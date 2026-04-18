import { IBufferTex, createBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { IModelShape } from "./GptModel";

export interface IP20ToyRuntime {
    shapeKey: string;
    inProjWeight: IBufferTex;
    inProjBias: IBufferTex;
    controls: IBufferTex;
    stateUpdate: IBufferTex;
    readoutWeight: IBufferTex;
    readoutBias: IBufferTex;
    readout: IBufferTex;
    residualOut: IBufferTex;
    trace: IP20ToyTrace;
}

export interface IP20ToyTrace {
    tokens: number[];
    inputMean: number[];
    gateMean: number[];
    thetaMean: number[];
    stateNorm: number[];
    readoutNorm: number[];
}

function shapeKey(shape: IModelShape) {
    return `${shape.B}:${shape.T}:${shape.C}`;
}

function sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
}

function deterministicWeight(row: number, col: number, seed: number, scale: number) {
    let a = Math.sin((row + 1) * 12.9898 + (col + 1) * 78.233 + seed * 37.719);
    let b = Math.cos((row + 1) * 4.127 + (col + 1) * 15.917 + seed * 9.31);
    return (a * 0.65 + b * 0.35) * scale;
}

function makeTex(gl: WebGL2RenderingContext, width: number, height: number, data: Float32Array) {
    let tex = createBufferTex(gl, width, height, 1);
    tex.localBuffer = data;
    writeToBufferTex(gl, tex, data);
    return tex;
}

function normRow(buf: Float32Array, row: number, width: number) {
    let sum = 0;
    for (let i = 0; i < width; i++) {
        let v = buf[row * width + i];
        sum += v * v;
    }
    return Math.sqrt(sum / width);
}

export function createOrUpdateP20ToyRuntime(
    gl: WebGL2RenderingContext,
    shape: IModelShape,
    prev: IP20ToyRuntime | null,
): IP20ToyRuntime {
    let key = shapeKey(shape);
    if (prev?.shapeKey === key) {
        return prev;
    }

    let { T, C } = shape;
    let controlWidth = C * 4;
    let invSqrtC = 1 / Math.sqrt(C);
    let tokens = [2, 1, 0, 1, 1, 2, 0, 2, 1, 0, 1].slice(0, T);
    while (tokens.length < T) {
        tokens.push((tokens.length * 2 + 1) % 3);
    }

    let input = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        let tok = tokens[t] + 1;
        for (let c = 0; c < C; c++) {
            input[t * C + c] =
                0.32 * Math.sin(tok * (c + 1) * 0.17) +
                0.24 * Math.cos((t + 1) * (c + 2) * 0.11) +
                0.10 * Math.sin((t + c + 1) * 0.29);
        }
    }

    let inProjWeight = new Float32Array(controlWidth * C);
    let inProjBias = new Float32Array(controlWidth);
    for (let row = 0; row < controlWidth; row++) {
        inProjBias[row] = 0.05 * Math.sin(row * 0.37);
        for (let col = 0; col < C; col++) {
            inProjWeight[row * C + col] = deterministicWeight(row, col, 0.7, 0.42);
        }
    }

    let controls = new Float32Array(T * controlWidth);
    for (let t = 0; t < T; t++) {
        for (let row = 0; row < controlWidth; row++) {
            let acc = inProjBias[row];
            for (let col = 0; col < C; col++) {
                acc += inProjWeight[row * C + col] * input[t * C + col] * invSqrtC;
            }
            controls[t * controlWidth + row] = acc;
        }
    }

    let stateUpdate = new Float32Array(T * controlWidth);
    let readoutWeight = new Float32Array(C * controlWidth);
    let readoutBias = new Float32Array(C);
    for (let row = 0; row < C; row++) {
        readoutBias[row] = 0.02 * Math.cos(row * 0.21);
        for (let col = 0; col < controlWidth; col++) {
            let lane = Math.floor(col / C);
            let laneScale = lane === 3 ? 0.42 : 0.13;
            readoutWeight[row * controlWidth + col] = deterministicWeight(row, col, 1.9, laneScale);
        }
    }

    let readout = new Float32Array(T * C);
    let residualOut = new Float32Array(T * C);
    let prevState = new Float32Array(C);
    let currState = new Float32Array(C);
    let rotated = new Float32Array(C);
    let candidate = new Float32Array(C);
    let gate = new Float32Array(C);
    let readGate = new Float32Array(C);

    let trace: IP20ToyTrace = {
        tokens,
        inputMean: [],
        gateMean: [],
        thetaMean: [],
        stateNorm: [],
        readoutNorm: [],
    };

    for (let t = 0; t < T; t++) {
        let controlBase = t * controlWidth;

        for (let c = 0; c < C; c += 2) {
            let theta0 = 0.8 * Math.tanh(controls[controlBase + C + c]);
            let theta1 = c + 1 < C ? 0.8 * Math.tanh(controls[controlBase + C + c + 1]) : theta0;
            let theta = (theta0 + theta1) * 0.5;
            let cosTheta = Math.cos(theta);
            let sinTheta = Math.sin(theta);
            let s0 = prevState[c];
            let s1 = c + 1 < C ? prevState[c + 1] : 0;
            rotated[c] = s0 * cosTheta - s1 * sinTheta;
            if (c + 1 < C) {
                rotated[c + 1] = s0 * sinTheta + s1 * cosTheta;
            }
        }

        let gateSum = 0;
        let thetaSum = 0;
        for (let c = 0; c < C; c++) {
            gate[c] = sigmoid(controls[controlBase + c]);
            candidate[c] = Math.tanh(controls[controlBase + 2 * C + c]);
            readGate[c] = sigmoid(controls[controlBase + 3 * C + c]);
            thetaSum += Math.abs(0.8 * Math.tanh(controls[controlBase + C + c]));
            gateSum += gate[c];
            currState[c] = gate[c] * rotated[c] + (1 - gate[c]) * candidate[c];

            stateUpdate[t * controlWidth + c] = currState[c];
            stateUpdate[t * controlWidth + C + c] = rotated[c];
            stateUpdate[t * controlWidth + 2 * C + c] = candidate[c];
            stateUpdate[t * controlWidth + 3 * C + c] = readGate[c] * currState[c];
        }

        for (let out = 0; out < C; out++) {
            let acc = readoutBias[out];
            for (let col = 0; col < controlWidth; col++) {
                acc += readoutWeight[out * controlWidth + col] * stateUpdate[t * controlWidth + col] * invSqrtC;
            }
            readout[t * C + out] = acc;
            residualOut[t * C + out] = input[t * C + out] + 0.45 * acc;
        }

        trace.inputMean.push(input.subarray(t * C, t * C + C).reduce((a, b) => a + Math.abs(b), 0) / C);
        trace.gateMean.push(gateSum / C);
        trace.thetaMean.push(thetaSum / C);
        trace.stateNorm.push(normRow(stateUpdate, t, C));
        trace.readoutNorm.push(normRow(readout, t, C));

        prevState.set(currState);
    }

    return {
        shapeKey: key,
        inProjWeight: makeTex(gl, C, controlWidth, inProjWeight),
        inProjBias: makeTex(gl, controlWidth, 1, inProjBias),
        controls: makeTex(gl, controlWidth, T, controls),
        stateUpdate: makeTex(gl, controlWidth, T, stateUpdate),
        readoutWeight: makeTex(gl, controlWidth, C, readoutWeight),
        readoutBias: makeTex(gl, 1, C, readoutBias),
        readout: makeTex(gl, C, T, readout),
        residualOut: makeTex(gl, C, T, residualOut),
        trace,
    };
}

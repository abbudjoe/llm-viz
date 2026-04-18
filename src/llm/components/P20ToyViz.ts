import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IProgramState } from "../Program";
import { RenderPhase } from "../render/sharedRender";
import { drawText, measureText } from "../render/fontRender";
import { addLine2, makeLineOpts } from "../render/lineRender";
import { addQuad } from "../render/triRender";
import { drawRoundedRect } from "./DataFlow";
import { drawLineRect } from "./ModelCard";

let panelBg = new Vec4(0.05, 0.11, 0.20, 0.88);
let panelLine = new Vec4(0.33, 0.63, 0.95, 0.95);
let panelText = new Vec4(0.88, 0.95, 1.0, 1.0);
let dimText = new Vec4(0.65, 0.78, 0.92, 1.0);
let tokenColor = new Vec4(0.35, 0.76, 1.0, 0.95);
let gateColor = new Vec4(0.35, 0.92, 0.56, 0.95);
let thetaColor = new Vec4(1.0, 0.78, 0.26, 0.95);
let stateColor = new Vec4(0.70, 0.52, 1.0, 0.95);
let readoutColor = new Vec4(1.0, 0.43, 0.43, 0.95);

function maxVal(values: number[]) {
    return values.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6);
}

function drawLabel(state: IProgramState, text: string, x: number, y: number, size = 13, color = panelText) {
    drawText(state.render.modelFontBuf, text, x, y, { color, size, mtx: new Mat4f() });
}

export function drawP20ToyScan(state: IProgramState, progress: number) {
    let toy = state.p20ToyRuntime;
    if (!toy) {
        return;
    }

    let prevPhase = state.render.sharedRender.activePhase;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;

    let mtx = new Mat4f();
    let width = 330;
    let height = 238;
    let margin = 18;
    let x = Math.max(20, state.render.size.x - width - 28);
    let y = 126;
    let tl = new Vec3(x, y, 0);
    let br = new Vec3(x + width, y + height, 0);

    drawRoundedRect(state.render, tl, br, panelBg, mtx, 12);
    drawLineRect(state.render, tl, br, makeLineOpts({ color: panelLine, thick: 1.2, n: new Vec3(0, 0, 1), mtx }));

    drawLabel(state, 'Live toy P20 recurrent scan', x + margin, y + 22, 15, panelText);
    drawLabel(state, 'packed controls -> rotary state -> readout', x + margin, y + 42, 11, dimText);

    let trace = toy.trace;
    let T = trace.tokens.length;
    let plotX = x + 86;
    let plotY = y + 68;
    let plotW = width - 110;
    let rowH = 27;
    let cellGap = 3;
    let cellW = (plotW - cellGap * (T - 1)) / T;
    let current = Math.min(T - 1, Math.max(0, Math.floor(progress * T)));

    let rows = [
        { label: 'x_t', values: trace.inputMean, scale: maxVal(trace.inputMean), color: tokenColor },
        { label: 'gate', values: trace.gateMean, scale: 1.0, color: gateColor },
        { label: 'theta', values: trace.thetaMean, scale: 0.8, color: thetaColor },
        { label: 'state', values: trace.stateNorm, scale: maxVal(trace.stateNorm), color: stateColor },
        { label: 'readout', values: trace.readoutNorm, scale: maxVal(trace.readoutNorm), color: readoutColor },
    ];

    let tokenText = trace.tokens.map(t => ['A', 'B', 'C'][t] ?? '?').join(' ');
    drawLabel(state, `tokens: ${tokenText}`, x + margin, y + height - 18, 11, dimText);

    for (let row = 0; row < rows.length; row++) {
        let r = rows[row];
        let rowY = plotY + row * rowH;
        drawLabel(state, r.label, x + margin, rowY + 12, 12, dimText);
        addLine2(
            state.render.lineRender,
            new Vec3(plotX, rowY + 14, 0),
            new Vec3(plotX + plotW, rowY + 14, 0),
            makeLineOpts({ color: new Vec4(1, 1, 1, 0.12), thick: 0.6, n: new Vec3(0, 0, 1), mtx }),
        );

        for (let t = 0; t < T; t++) {
            let value = Math.max(0.02, Math.min(1, Math.abs(r.values[t]) / r.scale));
            let barH = 18 * value;
            let bx = plotX + t * (cellW + cellGap);
            let by = rowY + 20 - barH;
            let color = t <= current ? r.color : r.color.mul(0.24);
            addQuad(
                state.render.triRender,
                new Vec3(bx, by, 0),
                new Vec3(bx + cellW, rowY + 20, 0),
                color,
                mtx,
            );
        }
    }

    let scanX = plotX + current * (cellW + cellGap) + cellW / 2;
    addLine2(
        state.render.lineRender,
        new Vec3(scanX, plotY - 10, 0),
        new Vec3(scanX, plotY + rows.length * rowH, 0),
        makeLineOpts({ color: new Vec4(1.0, 1.0, 1.0, 0.88), thick: 1.4, n: new Vec3(0, 0, 1), mtx }),
    );

    let currentText = `t=${current}  gate=${trace.gateMean[current].toFixed(2)}  state=${trace.stateNorm[current].toFixed(2)}`;
    let tw = measureText(state.render.modelFontBuf, currentText, { color: panelText, size: 11, mtx });
    drawLabel(state, currentText, x + width - margin - tw, y + height - 18, 11, panelText);

    state.render.sharedRender.activePhase = prevPhase;
}

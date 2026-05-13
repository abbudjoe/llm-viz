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
let carryColor = new Vec4(0.34, 0.54, 1.0, 0.95);
let rotatedColor = new Vec4(0.82, 0.63, 1.0, 0.95);
let candidateColor = new Vec4(1.0, 0.72, 0.34, 0.95);
let stateColor = new Vec4(1.0, 0.88, 0.18, 0.98);
let readoutColor = new Vec4(1.0, 0.43, 0.43, 0.95);
let warningColor = new Vec4(1.0, 0.52, 0.22, 0.98);
let mutedPanel = new Vec4(0.10, 0.20, 0.32, 0.74);
let faintLine = new Vec4(1.0, 1.0, 1.0, 0.20);

export type P2050MControllerFocus = 'projection' | 'state' | 'readout' | 'all';

const p20Controller50M = {
    params: '~50M',
    dModel: 448,
    layers: 8,
    heads: 8,
    headDim: 56,
    ffnDim: 1792,
    packedProjection: 1568,
    stateBlocks: 4,
    blockWidth: 112,
    rotaryPairs: 56,
    paddedPairs: 64,
    smallDModel: 128,
    smallPairTile: 16,
};

function maxVal(values: number[]) {
    return values.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6);
}

function drawLabel(state: IProgramState, text: string, x: number, y: number, size = 13, color = panelText) {
    drawText(state.render.modelFontBuf, text, x, y, { color, size, mtx: new Mat4f() });
}

function clamp01(v: number) {
    return Math.max(0, Math.min(1, v));
}

function focusAlpha(focus: P2050MControllerFocus, target: P2050MControllerFocus) {
    return focus === 'all' || focus === target ? 1.0 : 0.34;
}

function drawPanelLine(state: IProgramState, mtx: Mat4f, x0: number, y0: number, x1: number, y1: number, color = faintLine, thick = 1.0) {
    addLine2(
        state.render.lineRender,
        new Vec3(x0, y0, 0),
        new Vec3(x1, y1, 0),
        makeLineOpts({ color, thick, n: new Vec3(0, 0, 1), mtx }),
    );
}

function drawArrow(state: IProgramState, mtx: Mat4f, x0: number, y0: number, x1: number, y1: number, color: Vec4, thick = 1.2) {
    drawPanelLine(state, mtx, x0, y0, x1, y1, color, thick);

    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = Math.max(1e-6, Math.sqrt(dx * dx + dy * dy));
    let ux = dx / len;
    let uy = dy / len;
    let px = -uy;
    let py = ux;
    let head = 7;
    let wing = 4;

    drawPanelLine(state, mtx, x1, y1, x1 - ux * head + px * wing, y1 - uy * head + py * wing, color, thick);
    drawPanelLine(state, mtx, x1, y1, x1 - ux * head - px * wing, y1 - uy * head - py * wing, color, thick);
}

function drawBox(
    state: IProgramState,
    mtx: Mat4f,
    x: number,
    y: number,
    w: number,
    h: number,
    fill: Vec4,
    stroke: Vec4,
    label?: string,
    size = 12,
) {
    let tl = new Vec3(x, y, 0);
    let br = new Vec3(x + w, y + h, 0);
    drawRoundedRect(state.render, tl, br, fill, mtx, 7);
    drawLineRect(state.render, tl, br, makeLineOpts({ color: stroke, thick: 0.9, n: new Vec3(0, 0, 1), mtx }));
    if (label) {
        let tw = measureText(state.render.modelFontBuf, label, { color: panelText, size, mtx });
        drawLabel(state, label, x + (w - tw) * 0.5, y + (h - size) * 0.5 + 1, size, panelText);
    }
}

function drawMetricCard(state: IProgramState, mtx: Mat4f, x: number, y: number, w: number, title: string, value: string, color: Vec4) {
    drawBox(state, mtx, x, y, w, 42, mutedPanel, color.mul(0.78));
    drawLabel(state, title, x + 8, y + 13, 9.5, dimText);
    drawLabel(state, value, x + 8, y + 30, 13, panelText);
}

function drawSegment(
    state: IProgramState,
    mtx: Mat4f,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    color: Vec4,
    alpha: number,
) {
    addQuad(
        state.render.triRender,
        new Vec3(x, y, 0),
        new Vec3(x + w, y + h, 0),
        color.mul(alpha),
        mtx,
    );
    drawLineRect(state.render, new Vec3(x, y, 0), new Vec3(x + w, y + h, 0), makeLineOpts({ color: panelLine.mul(alpha), thick: 0.7, n: new Vec3(0, 0, 1), mtx }));
    if (w > 36) {
        drawLabel(state, label, x + 5, y + 13, 9, panelText.mul(Math.max(0.45, alpha)));
    }
}

export function drawP2050MController(state: IProgramState, progress: number, focus: P2050MControllerFocus = 'all') {
    let prevPhase = state.render.sharedRender.activePhase;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;

    progress = clamp01(progress);

    let mtx = new Mat4f();
    let width = Math.min(488, Math.max(420, state.render.size.x - 52));
    let height = 356;
    let x = 24;
    let y = Math.max(130, state.render.size.y - height - 26);
    let tl = new Vec3(x, y, 0);
    let br = new Vec3(x + width, y + height, 0);

    drawRoundedRect(state.render, tl, br, panelBg, mtx, 13);
    drawLineRect(state.render, tl, br, makeLineOpts({ color: panelLine, thick: 1.2, n: new Vec3(0, 0, 1), mtx }));

    drawLabel(state, '50M P20 / RGRP controller: current config', x + 18, y + 24, 15, panelText);
    drawLabel(state, 'scale-faithful inset; the 3D stack remains the toy walkthrough scaffold', x + 18, y + 43, 10.5, dimText);

    let cardY = y + 58;
    let cardGap = 8;
    let cardW = (width - 36 - cardGap * 3) / 4;
    drawMetricCard(state, mtx, x + 18, cardY, cardW, 'shape', `${p20Controller50M.layers}L x d${p20Controller50M.dModel}`, tokenColor);
    drawMetricCard(state, mtx, x + 18 + (cardW + cardGap), cardY, cardW, 'attention', `${p20Controller50M.heads} x ${p20Controller50M.headDim}`, gateColor);
    drawMetricCard(state, mtx, x + 18 + (cardW + cardGap) * 2, cardY, cardW, 'ffn seam', `d_ff ${p20Controller50M.ffnDim}`, candidateColor);
    drawMetricCard(state, mtx, x + 18 + (cardW + cardGap) * 3, cardY, cardW, 'params', p20Controller50M.params, readoutColor);

    let flowY = y + 123;
    let inputAlpha = focusAlpha(focus, 'projection');
    let stateAlpha = focusAlpha(focus, 'state');
    let readoutAlpha = focusAlpha(focus, 'readout');
    let pulse = 0.58 + 0.32 * Math.sin(progress * Math.PI * 2);

    drawBox(state, mtx, x + 18, flowY, 72, 38, tokenColor.mul(0.24 * inputAlpha), tokenColor.mul(inputAlpha), 'x_t R448', 11);
    drawArrow(state, mtx, x + 92, flowY + 19, x + 120, flowY + 19, tokenColor.mul(inputAlpha), 1.4);
    drawBox(state, mtx, x + 122, flowY, 86, 38, panelLine.mul(0.22 * inputAlpha), panelLine.mul(inputAlpha), 'W_in', 12);
    drawLabel(state, '448 -> 1568', x + 136, flowY + 33, 9.2, dimText.mul(inputAlpha));

    let splitX = x + 224;
    let splitY = flowY - 10;
    let splitW = width - (splitX - x) - 18;
    let dims = [448, 224, 448, 448];
    let labels = ['gate 448', 'theta 224', 'cand 448', 'read 448'];
    let colors = [gateColor, thetaColor, candidateColor, readoutColor];
    let sx = splitX;
    for (let i = 0; i < dims.length; i++) {
        let segW = splitW * dims[i] / p20Controller50M.packedProjection;
        drawSegment(state, mtx, sx, splitY, segW, 21, labels[i], colors[i], inputAlpha * (i === Math.floor(progress * 4) ? pulse : 0.72));
        sx += segW;
    }
    drawLabel(state, 'one packed projection creates every controller lane', splitX, splitY - 7, 9.5, dimText.mul(inputAlpha));

    let stateY = y + 190;
    let blockX = x + 106;
    let blockW = Math.min(54, (width - 238) / 4);
    let blockGap = 9;
    let blockH = 82;
    let stateRight = blockX + p20Controller50M.stateBlocks * blockW + (p20Controller50M.stateBlocks - 1) * blockGap;
    let currentBlock = Math.min(p20Controller50M.stateBlocks - 1, Math.floor(progress * p20Controller50M.stateBlocks));

    drawLabel(state, 'shared recurrent state highway', x + 18, stateY - 14, 11.5, stateColor.mul(stateAlpha));
    drawLabel(state, 's_t R448 = 4 block-diagonal groups', blockX, stateY - 14, 11.5, panelText.mul(stateAlpha));
    drawArrow(state, mtx, x + 26, stateY + 31, blockX - 10, stateY + 31, stateColor.mul(stateAlpha), 1.6);
    drawLabel(state, 's[t-1]', x + 26, stateY + 24, 10.5, stateColor.mul(stateAlpha));

    for (let i = 0; i < p20Controller50M.stateBlocks; i++) {
        let bx = blockX + i * (blockW + blockGap);
        let active = i === currentBlock ? pulse : 0.62;
        let stroke = i === currentBlock ? stateColor : panelLine;
        drawBox(state, mtx, bx, stateY, blockW, blockH, carryColor.mul(0.18 * stateAlpha * active), stroke.mul(stateAlpha), '', 10);
        drawLabel(state, `B${i}`, bx + 8, stateY + 18, 10.5, panelText.mul(stateAlpha));
        drawLabel(state, `${p20Controller50M.blockWidth}`, bx + 8, stateY + 36, 13, panelText.mul(stateAlpha * active));
        drawLabel(state, 'dims', bx + 8, stateY + 50, 8.5, dimText.mul(stateAlpha));

        let tickY = stateY + 62;
        for (let j = 0; j < 8; j++) {
            let tx = bx + 7 + j * Math.max(3.2, (blockW - 16) / 8);
            drawPanelLine(state, mtx, tx, tickY, tx, tickY + 10, thetaColor.mul(stateAlpha * (i === currentBlock ? 0.85 : 0.35)), 0.65);
        }
    }

    drawArrow(state, mtx, stateRight + 10, stateY + 31, x + width - 88, stateY + 31, readoutColor.mul(readoutAlpha), 1.5);
    drawBox(state, mtx, x + width - 82, stateY + 12, 58, 38, readoutColor.mul(0.18 * readoutAlpha), readoutColor.mul(readoutAlpha), 'readout', 10.5);
    drawLabel(state, 'back to residual stream', x + width - 124, stateY + 64, 10, dimText.mul(readoutAlpha));

    let loopY = stateY + blockH + 14;
    drawPanelLine(state, mtx, stateRight, stateY + blockH - 7, stateRight + 13, stateY + blockH - 7, stateColor.mul(stateAlpha), 1.5);
    drawPanelLine(state, mtx, stateRight + 13, stateY + blockH - 7, stateRight + 13, loopY, stateColor.mul(stateAlpha), 1.5);
    drawPanelLine(state, mtx, stateRight + 13, loopY, blockX - 22, loopY, stateColor.mul(stateAlpha), 1.5);
    drawArrow(state, mtx, blockX - 22, loopY, blockX - 22, stateY + 42, stateColor.mul(stateAlpha), 1.5);
    drawLabel(state, 's[t] is stored, then consumed by token t+1', blockX - 18, loopY + 14, 10, stateColor.mul(stateAlpha));

    drawLabel(state, 'per block: 56 rotary pairs -> Triton-padded pair tile 64', blockX, stateY + blockH + 31, 9.5, warningColor.mul(stateAlpha));

    let tileY = y + height - 70;
    drawPanelLine(state, mtx, x + 18, tileY - 16, x + width - 18, tileY - 16, faintLine, 0.8);
    drawLabel(state, 'why width hurts the current controller', x + 18, tileY - 1, 11.5, warningColor);

    let smallX = x + 20;
    let largeX = x + 226;
    drawBox(state, mtx, smallX, tileY + 12, 34, 34, gateColor.mul(0.22), gateColor.mul(0.8));
    drawLabel(state, `10M d${p20Controller50M.smallDModel}: 4 x ${p20Controller50M.smallPairTile}x${p20Controller50M.smallPairTile}`, smallX + 46, tileY + 28, 10.5, dimText);
    drawBox(state, mtx, largeX, tileY + 3, 52, 52, warningColor.mul(0.22), warningColor);
    drawLabel(state, `50M d${p20Controller50M.dModel}: 4 x ${p20Controller50M.paddedPairs}x${p20Controller50M.paddedPairs}`, largeX + 64, tileY + 24, 10.5, panelText);
    drawLabel(state, '16x scan-tile proxy, not a mere 3.5x width tax', largeX + 64, tileY + 43, 10.2, warningColor);

    state.render.sharedRender.activePhase = prevPhase;
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
    let height = 302;
    let margin = 18;
    let x = Math.max(20, state.render.size.x - width - 28);
    let y = 126;
    let tl = new Vec3(x, y, 0);
    let br = new Vec3(x + width, y + height, 0);

    drawRoundedRect(state.render, tl, br, panelBg, mtx, 12);
    drawLineRect(state.render, tl, br, makeLineOpts({ color: panelLine, thick: 1.2, n: new Vec3(0, 0, 1), mtx }));

    drawLabel(state, 'Toy recurrent scan: mechanism values', x + margin, y + 22, 15, panelText);
    drawLabel(state, 'computed browser toy, not the 50M tensor shape', x + margin, y + 42, 11, dimText);

    let trace = toy.trace;
    let T = trace.tokens.length;
    let plotX = x + 86;
    let plotY = y + 68;
    let plotW = width - 110;
    let rowH = 24;
    let cellGap = 3;
    let cellW = (plotW - cellGap * (T - 1)) / T;
    let current = Math.min(T - 1, Math.max(0, Math.floor(progress * T)));

    let rows = [
        { label: 'x_t', values: trace.inputMean, scale: maxVal(trace.inputMean), color: tokenColor },
        { label: 'gate', values: trace.gateMean, scale: 1.0, color: gateColor },
        { label: 'theta', values: trace.thetaMean, scale: 0.8, color: thetaColor },
        { label: 'carry in', values: trace.prevStateNorm, scale: maxVal(trace.stateNorm), color: carryColor },
        { label: 'rotate', values: trace.rotatedNorm, scale: maxVal(trace.stateNorm), color: rotatedColor },
        { label: 'cand', values: trace.candidateNorm, scale: maxVal(trace.candidateNorm), color: candidateColor },
        { label: 'new carry', values: trace.stateNorm, scale: maxVal(trace.stateNorm), color: stateColor },
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

    function cellCenter(t: number, row: number) {
        return new Vec3(
            plotX + t * (cellW + cellGap) + cellW / 2,
            plotY + row * rowH + 11,
            0,
        );
    }

    if (current > 0) {
        let lineOpts = makeLineOpts({ color: stateColor, thick: 1.4, n: new Vec3(0, 0, 1), mtx });
        addLine2(state.render.lineRender, cellCenter(current - 1, 6), cellCenter(current, 3), lineOpts);
        addLine2(state.render.lineRender, cellCenter(current, 3), cellCenter(current, 4), makeLineOpts({ color: rotatedColor, thick: 1.0, n: new Vec3(0, 0, 1), mtx }));
        addLine2(state.render.lineRender, cellCenter(current, 4), cellCenter(current, 6), lineOpts);
        drawLabel(state, 'carry feeds next token update', plotX, y + height - 38, 10.5, stateColor);
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

import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { addSourceDestCurveLine, drawTextOnModel, splitGridForHighlight, TextAlignHoriz, TextAlignVert } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { drawP20ToyScan } from "../components/P20ToyViz";
import { cellPosition, IBlkDef } from "../GptModelLayout";
import { IP20ModelLayout } from "../P20ModelLayout";
import { addLine } from "../render/lineRender";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, ITimeInfo, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

function isAnimating(t: ITimeInfo) {
    return t.active && t.t > 0 && t.t < 1;
}

function highlightRailCell(layout: IP20ModelLayout, rail: NonNullable<IP20ModelLayout['p20StateRails']>[number], tokenIdx: number, channelIdx: number, strength: number) {
    let tokenSlice = splitGridForHighlight(layout, rail, Dim.X, tokenIdx);
    let cell = tokenSlice ? splitGridForHighlight(layout, tokenSlice, Dim.Y, channelIdx) : null;
    if (cell) {
        cell.highlight = Math.max(cell.highlight, strength);
    }
}

function railCellCenter(layout: IP20ModelLayout, rail: NonNullable<IP20ModelLayout['p20StateRails']>[number], tokenIdx: number, channelIdx: number) {
    return new Vec3(
        cellPosition(layout, rail, Dim.X, tokenIdx) + layout.cell * 0.5,
        cellPosition(layout, rail, Dim.Y, channelIdx) + layout.cell * 0.5,
        rail.z + rail.dz + layout.margin * 0.45,
    );
}

function drawStateRailFlow(layout: IP20ModelLayout, state: IWalkthroughArgs['state'], rail: NonNullable<IP20ModelLayout['p20StateRails']>[number], prevTokenIdx: number, tokenIdx: number, channelIdx: number) {
    if (prevTokenIdx === tokenIdx) {
        return;
    }

    let highwayColor = new Vec4(1.0, 0.88, 0.18, 0.98);
    let prev = railCellCenter(layout, rail, prevTokenIdx, channelIdx);
    let curr = railCellCenter(layout, rail, tokenIdx, channelIdx);
    let lift = new Vec3(0, -layout.margin * 0.12, layout.margin * 0.18);

    addLine(state.render.lineRender, 8, highwayColor, prev, prev.add(lift));
    addLine(state.render.lineRender, 8, highwayColor, prev.add(lift), curr.add(lift));
    addLine(state.render.lineRender, 8, highwayColor, curr.add(lift), curr);

    drawTextOnModel(state.render, 'shared state highway: s[t-1] -> s[t]', new Vec3(rail.x + rail.dx * 0.5, rail.y - layout.margin * 0.28, rail.z + rail.dz + 2), {
        color: highwayColor,
        size: 2.6,
        align: TextAlignHoriz.Center,
        valign: TextAlignVert.Bottom,
    });
}

function drawStateHighwayUpdate(layout: IP20ModelLayout, state: IWalkthroughArgs['state'], block: IP20ModelLayout['blocks'][number], rail: NonNullable<IP20ModelLayout['p20StateRails']>[number], progress: number) {
    let T = layout.shape.T;
    let C = layout.shape.C;
    let tokenIdx = Math.min(T - 1, Math.max(0, Math.floor(progress * T)));
    let prevTokenIdx = Math.max(0, tokenIdx - 1);
    let channelIdx = Math.min(8, C - 1);
    let stateIn = new Vec4(0.36, 0.50, 1.00, 0.95);
    let stateOut = new Vec4(0.18, 0.86, 0.42, 0.95);

    rail.highlight = Math.max(rail.highlight, 1.0);
    block.mlpAct.highlight = Math.max(block.mlpAct.highlight, 1.0);

    highlightRailCell(layout, rail, prevTokenIdx, channelIdx, 0.8);
    highlightRailCell(layout, rail, tokenIdx, channelIdx, 1.0);
    drawStateRailFlow(layout, state, rail, prevTokenIdx, tokenIdx, channelIdx);

    addSourceDestCurveLine(state.render, layout, rail, block.mlpAct, new Vec3(prevTokenIdx, channelIdx, 0), new Vec3(channelIdx, tokenIdx, 0), stateIn);
    addSourceDestCurveLine(state.render, layout, block.mlpAct, rail, new Vec3(channelIdx, tokenIdx, 0), new Vec3(tokenIdx, channelIdx, 0), stateOut);

    drawTextOnModel(state.render, 'read previous state', new Vec3(rail.x, rail.y - layout.margin * 0.54, rail.z + rail.dz + 1), {
        color: stateIn,
        size: 2.4,
        align: TextAlignHoriz.Left,
        valign: TextAlignVert.Bottom,
    });
    drawTextOnModel(state.render, 'write updated state', new Vec3(rail.x + rail.dx, rail.y + rail.dy + layout.margin * 0.2, rail.z + rail.dz + 1), {
        color: stateOut,
        size: 2.2,
        align: TextAlignHoriz.Right,
        valign: TextAlignVert.Top,
    });
}

function drawStateHighwayReadout(layout: IP20ModelLayout, state: IWalkthroughArgs['state'], block: IP20ModelLayout['blocks'][number], rail: NonNullable<IP20ModelLayout['p20StateRails']>[number], progress: number) {
    let T = layout.shape.T;
    let C = layout.shape.C;
    let tokenIdx = Math.min(T - 1, Math.max(0, Math.floor(progress * T)));
    let channelIdx = Math.min(8, C - 1);
    let readColor = new Vec4(1.00, 0.45, 0.38, 0.95);

    rail.highlight = Math.max(rail.highlight, 1.0);
    block.mlpResult.highlight = Math.max(block.mlpResult.highlight, 1.0);
    highlightRailCell(layout, rail, tokenIdx, channelIdx, 1.0);

    addSourceDestCurveLine(state.render, layout, rail, block.mlpResult, new Vec3(tokenIdx, channelIdx, 0), new Vec3(tokenIdx, channelIdx, 0), readColor);
    drawTextOnModel(state.render, 'state readout returns to residual stream', new Vec3(rail.x + rail.dx, block.mlpResult.y - layout.margin * 0.3, rail.z + rail.dz + 1), {
        color: readColor,
        size: 2.2,
        align: TextAlignHoriz.Right,
        valign: TextAlignVert.Bottom,
    });
}

export function walkthrough10_P20(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout: rawLayout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.P20_Detail_RecurrentControl) {
        return;
    }

    let layout = rawLayout as IP20ModelLayout;
    let blockIdx = Math.min(1, layout.blocks.length - 1);
    let block = layout.blocks[blockIdx];
    let stateRail = layout.p20StateRails?.[Math.max(0, blockIdx - 1)] ?? layout.p20StateRails?.[0];
    let p20Blocks = [
        block.ln2.lnResid,
        block.mlpFcWeight,
        block.mlpFcBias,
        block.mlpFc,
        block.mlpAct,
        block.mlpProjWeight,
        block.mlpProjBias,
        block.mlpResult,
        block.mlpResidual,
        stateRail,
    ].filter((cube): cube is IBlkDef => Boolean(cube));

    wt.dimHighlightBlocks = p20Blocks;
    setInitialCamera(state, new Vec3(-160.000, 0.000, -720.000), new Vec3(289.500, -8.000, 3.600));

    commentary(wt)`
P20 is not a totally different animal from nanoGPT. It keeps the same token embedding, residual stream,
layer norms, self-attention, projection, final layer norm, and output head.

The architectural change happens at the feed-forward seam. In nanoGPT this is a plain expand, GELU,
and project MLP. In the P20 ablation, that seam becomes a ${c_blockRef('rotary gated recurrent state update', block.mlpAct)}
with a compact state highway.

This chapter now runs a tiny browser-side P20 calculation for the visualized seam. It is still a toy
mechanism, not the trained 9.87M research checkpoint, but the gates, angles, state, readout, and residual
values shown here are computed live from the toy recurrence.
`;
    breakAfter();

    let t0_focus = afterTime(null, 1.0, 0.4);
    if (t0_focus.active) {
        for (let cube of p20Blocks) {
            cube.highlight = Math.max(cube.highlight, 0.9 * t0_focus.t);
        }
        block.mlpLabel.visible = t0_focus.t;
    }

    cleanup(t0_focus);

    commentary(wt)`
The incoming ${c_dimRef('C', DimStyle.C)}-channel residual vector first passes through one packed projection.
That projection is deliberately doing more than a normal MLP input matrix: it creates update gates,
rotary angles, candidate state values, and readout gates in one place.
`;
    breakAfter();

    let t1_projection = afterTime(null, 3.0, 0.3);
    let showProjectionOverlay = isAnimating(t1_projection);
    if (t1_projection.active) {
        block.mlpFcWeight.highlight = 0.9;
        block.mlpFcBias.highlight = 0.5;
        block.mlpFc.highlight = 0.9;
    }

    commentary(wt)`
Those projected controls update a recurrent state. The key distinction from the vanilla MLP is that the
operation can carry information forward through a compact state, rather than treating every token column
as a completely independent feed-forward calculation.
`;
    breakAfter();

    let t2_update = afterTime(null, 3.0, 0.3);
    showProjectionOverlay &&= !t2_update.active;
    let showUpdateOverlay = isAnimating(t2_update);
    if (t2_update.active) {
        block.mlpAct.highlight = 1.0;
        block.mlpResult.highlight = 0.6;
        if (stateRail) {
            stateRail.highlight = Math.max(stateRail.highlight, 0.8);
        }
    }

    commentary(wt)`
After the recurrent update, P20 emits a readout back into the normal residual stream. That is why the
rest of the model can stay transformer-like: attention still does explicit token mixing, and the output
head still sees an ordinary residual vector.
`;
    breakAfter();

    let t3_readout = afterTime(null, 3.0, 0.4);
    showUpdateOverlay &&= !t3_readout.active;
    let showReadoutOverlay = isAnimating(t3_readout);
    if (t3_readout.active) {
        block.mlpProjWeight.highlight = 0.8;
        block.mlpResult.highlight = 1.0;
        block.mlpResidual.highlight = 0.9;
        if (stateRail) {
            stateRail.highlight = Math.max(stateRail.highlight, 0.8);
        }
    }

    commentary(wt)`
In short: P20 is closest to nanoGPT at the shell, and different at the FFN-side computation. The ablation
asks whether a small recurrent-control primitive can buy some of the useful depth or state behavior that
a plain MLP does not provide.
`;
    breakAfter();

    let t4_processAll = afterTime(null, 5.0);
    showReadoutOverlay &&= !t4_processAll.active;
    let showFullScanOverlay = isAnimating(t4_processAll);

    if (showProjectionOverlay) {
        drawDependences(state, block.mlpFc, new Vec3(3, 8, 0));
        drawDataFlow(state, block.mlpFc, new Vec3(3, 8, 0), new Vec3(20, -12, 0));
        drawP20ToyScan(state, t1_projection.t * 0.25);
    }
    if (showUpdateOverlay) {
        drawDataFlow(state, block.mlpAct, new Vec3(8, 16, 0), new Vec3(block.mlpAct.cx / 2, -16, 0));
        if (stateRail) {
            drawStateHighwayUpdate(layout, state, block, stateRail, 0.25 + t2_update.t * 0.35);
        }
        drawP20ToyScan(state, 0.25 + t2_update.t * 0.35);
    }
    if (showReadoutOverlay) {
        drawDataFlow(state, block.mlpResult, new Vec3(3, 8, 0), new Vec3(block.mlpResult.cx / 2, -16, 0));
        if (stateRail) {
            drawStateHighwayReadout(layout, state, block, stateRail, 0.6 + t3_readout.t * 0.25);
        }
        drawP20ToyScan(state, 0.6 + t3_readout.t * 0.25);
    }

    if (t4_processAll.t > 0) {
        let prevInfo = startProcessBefore(state, block.ln2.lnResid);
        processUpTo(state, t4_processAll, block.mlpResidual, prevInfo);
        for (let cube of p20Blocks) {
            cube.highlight = Math.max(cube.highlight, 0.35);
        }
        if (stateRail && showFullScanOverlay) {
            drawStateHighwayUpdate(layout, state, block, stateRail, t4_processAll.t);
            drawStateHighwayReadout(layout, state, block, stateRail, t4_processAll.t);
            drawP20ToyScan(state, t4_processAll.t);
        }
    }
}

import { Vec3 } from "@/src/utils/vector";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { drawP20ToyScan } from "../components/P20ToyViz";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough10_P20(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.P20_Detail_RecurrentControl) {
        return;
    }

    let blockIdx = Math.min(1, layout.blocks.length - 1);
    let block = layout.blocks[blockIdx];
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
    ];

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
    if (t1_projection.active) {
        block.mlpFcWeight.highlight = 0.9;
        block.mlpFcBias.highlight = 0.5;
        block.mlpFc.highlight = 0.9;
        drawDependences(state, block.mlpFc, new Vec3(3, 8, 0));
        drawDataFlow(state, block.mlpFc, new Vec3(3, 8, 0), new Vec3(20, -12, 0));
        drawP20ToyScan(state, t1_projection.t * 0.25);
    }

    commentary(wt)`
Those projected controls update a recurrent state. The key distinction from the vanilla MLP is that the
operation can carry information forward through a compact state, rather than treating every token column
as a completely independent feed-forward calculation.
`;
    breakAfter();

    let t2_update = afterTime(null, 3.0, 0.3);
    if (t2_update.active) {
        block.mlpAct.highlight = 1.0;
        block.mlpResult.highlight = 0.6;
        drawDataFlow(state, block.mlpAct, new Vec3(8, 16, 0), new Vec3(block.mlpAct.cx / 2, -16, 0));
        drawP20ToyScan(state, 0.25 + t2_update.t * 0.35);
    }

    commentary(wt)`
After the recurrent update, P20 emits a readout back into the normal residual stream. That is why the
rest of the model can stay transformer-like: attention still does explicit token mixing, and the output
head still sees an ordinary residual vector.
`;
    breakAfter();

    let t3_readout = afterTime(null, 3.0, 0.4);
    if (t3_readout.active) {
        block.mlpProjWeight.highlight = 0.8;
        block.mlpResult.highlight = 1.0;
        block.mlpResidual.highlight = 0.9;
        drawDataFlow(state, block.mlpResult, new Vec3(3, 8, 0), new Vec3(block.mlpResult.cx / 2, -16, 0));
        drawP20ToyScan(state, 0.6 + t3_readout.t * 0.25);
    }

    commentary(wt)`
In short: P20 is closest to nanoGPT at the shell, and different at the FFN-side computation. The ablation
asks whether a small recurrent-control primitive can buy some of the useful depth or state behavior that
a plain MLP does not provide.
`;
    breakAfter();

    let t4_processAll = afterTime(null, 5.0);
    if (t4_processAll.t > 0) {
        let prevInfo = startProcessBefore(state, block.ln2.lnResid);
        processUpTo(state, t4_processAll, block.mlpResidual, prevInfo);
        for (let cube of p20Blocks) {
            cube.highlight = Math.max(cube.highlight, 0.35);
        }
        drawP20ToyScan(state, t4_processAll.t);
    }
}

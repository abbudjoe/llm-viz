import { Mat4f } from "@/src/utils/matrix";
import { Vec3 } from "@/src/utils/vector";
import { IGptModelLink, IModelShape } from "./GptModel";
import { IBufferTex } from "@/src/utils/renderPhases";
import { genGptModelLayout, BlKDepSpecial, IBlkAccess, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { IP20ToyRuntime } from "./P20ToyRuntime";
import { DimStyle } from "./walkthrough/WalkthroughTools";

function cloneVisualBlock(base: IBlkDef, overrides: Partial<IBlkDef>): IBlkDef {
    return {
        ...base,
        access: undefined,
        deps: undefined,
        localMtx: new Mat4f(),
        rangeOffsetsX: undefined,
        rangeOffsetsY: undefined,
        rangeOffsetsZ: undefined,
        subs: undefined,
        idx: -1,
        ...overrides,
    };
}

function ensure4(a: number[]) {
    return a.length === 4 ? a : [...a, 0];
}

function makeAccess(src: IBufferTex, x: number[], y: number[], scale = 1.0): IBlkAccess {
    return {
        src,
        channel: 'r',
        scale,
        mat: Mat4f.fromColMajor([...ensure4(x), ...ensure4(y), 0, 0, 0, 0, 0, 0, 0, 0]),
    };
}

function attachP20ToyRuntime(block: IGptModelLayout['blocks'][number], toy: IP20ToyRuntime) {
    block.mlpFcWeight.access = makeAccess(toy.inProjWeight, [0, 1, 0], [1, 0, 0], 3.0);
    block.mlpFcBias.access = makeAccess(toy.inProjBias, [1, 0, 0], [0, 0, 0], 4.0);
    block.mlpFc.access = makeAccess(toy.controls, [1, 0, 0], [0, 1, 0], 1.6);
    block.mlpAct.access = makeAccess(toy.stateUpdate, [1, 0, 0], [0, 1, 0], 2.1);
    block.mlpProjWeight.access = makeAccess(toy.readoutWeight, [1, 0, 0], [0, 1, 0], 4.0);
    block.mlpProjBias.access = makeAccess(toy.readoutBias, [0, 0, 0], [0, 1, 0], 8.0);
    block.mlpResult.access = makeAccess(toy.readout, [0, 1, 0], [1, 0, 0], 4.0);
    block.mlpResidual.access = makeAccess(toy.residualOut, [0, 1, 0], [1, 0, 0], 2.5);

    if (block.mlpFc.deps) {
        block.mlpFc.deps.special = BlKDepSpecial.P20PackedProjection;
    }
    if (block.mlpAct.deps) {
        block.mlpAct.deps.special = BlKDepSpecial.P20StateUpdate;
    }
    if (block.mlpResult.deps) {
        block.mlpResult.deps.special = BlKDepSpecial.P20Readout;
    }
    if (block.mlpResidual.deps) {
        block.mlpResidual.deps.special = BlKDepSpecial.P20ResidualMix;
    }
}

function retitleRecurrentSlot(block: IGptModelLayout['blocks'][number], toy?: IP20ToyRuntime | null) {
    block.mlpFcWeight.name = 'Packed P20 In-Projection';
    block.mlpFcBias.name = 'P20 Control Bias';
    block.mlpFc.name = 'Gate / Angle / Candidate Slots';
    block.mlpAct.name = 'Rotary Gated State Update';
    block.mlpProjWeight.name = 'P20 Readout Weights';
    block.mlpProjBias.name = 'P20 Readout Bias';
    block.mlpResult.name = 'P20 Recurrent Readout';
    block.mlpResidual.name = 'P20 Residual Mix';

    for (let cube of [
        block.mlpFcWeight,
        block.mlpFcBias,
        block.mlpFc,
        block.mlpAct,
        block.mlpProjWeight,
        block.mlpProjBias,
        block.mlpResult,
        block.mlpResidual,
    ]) {
        cube.highlight = 0.75;
    }

    for (let head of block.heads) {
        head.attnMtx.highlight = 0.25;
        head.attnMtxSm.highlight = 0.25;
        head.vOutBlock.highlight = 0.2;
    }
    block.attnOut.highlight = 0.2;
    block.attnResidual.highlight = 0.2;

    if (toy) {
        attachP20ToyRuntime(block, toy);
    }
}

export type IP20ModelLayout = IGptModelLayout & {
    p20StateRails?: IBlkDef[];
};

function makeStateRail(
    layout: IGptModelLayout,
    block: IGptModelLayout['blocks'][number],
    idx: number,
    toy?: IP20ToyRuntime | null,
): IBlkDef {
    let top = block.ln1.lnResid.y;
    let railCellsX = toy ? layout.shape.T : 3;
    let railCellsY = toy ? layout.shape.C : Math.max(1, Math.round((block.mlpResidual.y + block.mlpResidual.dy - top) / layout.cell));
    let railCellsZ = Math.max(1, layout.shape.B);

    return cloneVisualBlock(block.mlpResidual, {
        t: 'i',
        name: idx === 0 ? 'Shared P20 Recurrent State Highway' : 'P20 State Carry',
        x: block.mlpResidual.x + block.mlpResidual.dx + layout.margin * 0.45,
        y: top,
        z: block.mlpResidual.z - layout.margin * 0.35,
        dx: railCellsX * layout.cell,
        dy: railCellsY * layout.cell,
        dz: railCellsZ * layout.cell,
        cx: railCellsX,
        cy: railCellsY,
        cz: railCellsZ,
        access: toy ? makeAccess(toy.stateUpdate, [0, 1, 0], [1, 0, 0], 2.1) : undefined,
        dimX: toy ? DimStyle.T : DimStyle.C,
        dimY: toy ? DimStyle.C : DimStyle.n_layers,
        highlight: 1.0,
        opacity: 0.92,
        small: false,
    });
}

function applyP20VisualTreatment(layout: IGptModelLayout, toy?: IP20ToyRuntime | null) {
    let p20Start = Math.max(1, Math.floor(layout.blocks.length * 0.25));
    let p20End = Math.max(p20Start + 1, Math.ceil(layout.blocks.length * 0.75));
    let p20Blocks = layout.blocks.slice(p20Start, p20End);

    let stateRails = p20Blocks.map((block, i) => {
        retitleRecurrentSlot(block, toy);
        return makeStateRail(layout, block, i, toy);
    });

    for (let block of layout.blocks.slice(0, p20Start)) {
        block.mlpResidual.name = 'Prelude Transformer Residual';
    }

    for (let block of layout.blocks.slice(p20End)) {
        block.mlpResidual.name = 'Coda Transformer Residual';
    }

    layout.cubes.push(...stateRails);
    (layout as IP20ModelLayout).p20StateRails = stateRails;
    layout.cubes.forEach((cube, idx) => {
        cube.idx = idx;
    });
}

export function genP20ModelLayout(
    shape: IModelShape,
    gptGpuModel: IGptModelLink | null = null,
    offset: Vec3 = new Vec3(0, 0, 0),
    toy?: IP20ToyRuntime | null,
) {
    let layout = genGptModelLayout(shape, gptGpuModel, offset);
    applyP20VisualTreatment(layout, toy);
    return layout as IP20ModelLayout;
}

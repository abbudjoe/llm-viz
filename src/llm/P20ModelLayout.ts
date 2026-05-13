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
    p20LoopChamber?: IBlkDef;
    p20Controller?: IBlkDef;
    p20ScanHotspot?: IBlkDef;
};

export type P20LayoutMode = 'toy-walkthrough' | 'parcae-rgrp-control';

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

function blockBounds(blocks: IGptModelLayout['blocks']) {
    let cubes = blocks.flatMap(block => block.cubes);
    let minX = Math.min(...cubes.map(cube => cube.x));
    let minY = Math.min(...cubes.map(cube => cube.y));
    let minZ = Math.min(...cubes.map(cube => cube.z));
    let maxX = Math.max(...cubes.map(cube => cube.x + cube.dx));
    let maxY = Math.max(...cubes.map(cube => cube.y + cube.dy));
    let maxZ = Math.max(...cubes.map(cube => cube.z + cube.dz));
    return { minX, minY, minZ, maxX, maxY, maxZ };
}

function addParcaeRgrpTopologyVisuals(layout: IGptModelLayout, p20Blocks: IGptModelLayout['blocks']) {
    let bounds = blockBounds(p20Blocks);
    let firstBlock = p20Blocks[0];
    let cell = layout.cell;
    let margin = layout.margin;
    let loopPad = margin * 0.55;

    let chamber = cloneVisualBlock(firstBlock.mlpResidual, {
        t: 'a',
        name: 'Parcae Loop Chamber: middle layers 3-4 reused twice',
        x: bounds.minX - loopPad,
        y: bounds.minY - loopPad,
        z: bounds.minZ - loopPad * 0.16,
        dx: bounds.maxX - bounds.minX + loopPad * 2,
        dy: bounds.maxY - bounds.minY + loopPad * 2,
        dz: bounds.maxZ - bounds.minZ + loopPad * 0.32,
        cx: Math.max(1, Math.round((bounds.maxX - bounds.minX + loopPad * 2) / cell)),
        cy: Math.max(1, Math.round((bounds.maxY - bounds.minY + loopPad * 2) / cell)),
        cz: Math.max(1, Math.round((bounds.maxZ - bounds.minZ + loopPad * 0.32) / cell)),
        dimX: DimStyle.C,
        dimY: DimStyle.n_layers,
        highlight: 1.0,
        opacity: 0.22,
        small: false,
    });

    let controller = cloneVisualBlock(firstBlock.mlpResidual, {
        t: 'w',
        name: 'Full-width RGRP Controller: scan over loop input at d_model=448',
        x: bounds.maxX + margin * 0.65,
        y: bounds.minY,
        z: firstBlock.mlpResidual.z,
        dx: Math.max(8 * cell, layout.shape.T * cell * 0.035),
        dy: bounds.maxY - bounds.minY,
        dz: Math.max(2 * cell, layout.shape.B * cell),
        cx: 4,
        cy: layout.shape.C,
        cz: Math.max(1, layout.shape.B),
        dimX: DimStyle.C,
        dimY: DimStyle.C,
        highlight: 1.0,
        opacity: 0.92,
        small: false,
    });

    let stateHighway = cloneVisualBlock(firstBlock.mlpResidual, {
        t: 'i',
        name: 'RGRP Recurrent State Highway: s[t-1] -> s[t] across all tokens',
        x: controller.x + controller.dx + margin * 0.22,
        y: bounds.minY,
        z: controller.z - margin * 0.10,
        dx: Math.max(4 * cell, layout.shape.T * cell * 0.018),
        dy: bounds.maxY - bounds.minY,
        dz: controller.dz,
        cx: 4,
        cy: layout.shape.C,
        cz: Math.max(1, layout.shape.B),
        dimX: DimStyle.T,
        dimY: DimStyle.C,
        highlight: 1.0,
        opacity: 0.95,
        small: false,
    });

    let hotspot = cloneVisualBlock(firstBlock.mlpResidual, {
        t: 'a',
        name: 'Speed Hotspot: 4 blocks x 64x64 padded rotary state tiles',
        x: stateHighway.x + stateHighway.dx + margin * 0.22,
        y: bounds.minY + (bounds.maxY - bounds.minY) * 0.18,
        z: controller.z,
        dx: Math.max(6 * cell, layout.shape.T * cell * 0.025),
        dy: (bounds.maxY - bounds.minY) * 0.64,
        dz: controller.dz,
        cx: 4,
        cy: 64,
        cz: 64,
        dimX: DimStyle.C,
        dimY: DimStyle.C,
        highlight: 1.0,
        opacity: 0.88,
        small: false,
    });

    layout.cubes.push(chamber, controller, stateHighway, hotspot);
    (layout as IP20ModelLayout).p20LoopChamber = chamber;
    (layout as IP20ModelLayout).p20Controller = controller;
    (layout as IP20ModelLayout).p20ScanHotspot = hotspot;
}

function applyP20VisualTreatment(layout: IGptModelLayout, toy?: IP20ToyRuntime | null, mode: P20LayoutMode = 'toy-walkthrough') {
    let p20Start = mode === 'parcae-rgrp-control' && layout.blocks.length >= 8
        ? 3
        : Math.max(1, Math.floor(layout.blocks.length * 0.25));
    let p20End = mode === 'parcae-rgrp-control' && layout.blocks.length >= 8
        ? 5
        : Math.max(p20Start + 1, Math.ceil(layout.blocks.length * 0.75));
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

    if (mode === 'parcae-rgrp-control') {
        addParcaeRgrpTopologyVisuals(layout, p20Blocks);
    }

    layout.cubes.forEach((cube, idx) => {
        cube.idx = idx;
    });
}

export function genP20ModelLayout(
    shape: IModelShape,
    gptGpuModel: IGptModelLink | null = null,
    offset: Vec3 = new Vec3(0, 0, 0),
    toy?: IP20ToyRuntime | null,
    mode: P20LayoutMode = 'toy-walkthrough',
) {
    let layout = genGptModelLayout(shape, gptGpuModel, offset);
    applyP20VisualTreatment(layout, toy, mode);
    return layout as IP20ModelLayout;
}

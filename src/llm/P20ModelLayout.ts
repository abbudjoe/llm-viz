import { Mat4f } from "@/src/utils/matrix";
import { Vec3 } from "@/src/utils/vector";
import { IGptModelLink, IModelShape } from "./GptModel";
import { genGptModelLayout, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { DimStyle } from "./walkthrough/WalkthroughTools";

const P20_WEIGHT_COUNT = 9_870_000;

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

function retitleRecurrentSlot(block: IGptModelLayout['blocks'][number]) {
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
}

function makeStateRail(layout: IGptModelLayout, block: IGptModelLayout['blocks'][number], idx: number): IBlkDef {
    let top = block.ln1.lnResid.y;
    let bottom = block.mlpResidual.y + block.mlpResidual.dy;
    let railCellsY = Math.max(1, Math.round((bottom - top) / layout.cell));
    let railCellsX = 3;
    let railCellsZ = Math.max(2, Math.round(layout.shape.B + 1));

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
        dimX: DimStyle.C,
        dimY: DimStyle.n_layers,
        highlight: 1.0,
        opacity: 0.92,
        small: false,
    });
}

function applyP20VisualTreatment(layout: IGptModelLayout) {
    layout.weightCount = P20_WEIGHT_COUNT;

    let p20Start = Math.max(1, Math.floor(layout.blocks.length * 0.25));
    let p20End = Math.max(p20Start + 1, Math.ceil(layout.blocks.length * 0.75));
    let p20Blocks = layout.blocks.slice(p20Start, p20End);

    let stateRails = p20Blocks.map((block, i) => {
        retitleRecurrentSlot(block);
        return makeStateRail(layout, block, i);
    });

    for (let block of layout.blocks.slice(0, p20Start)) {
        block.mlpResidual.name = 'Prelude Transformer Residual';
    }

    for (let block of layout.blocks.slice(p20End)) {
        block.mlpResidual.name = 'Coda Transformer Residual';
    }

    layout.cubes.push(...stateRails);
    layout.cubes.forEach((cube, idx) => {
        cube.idx = idx;
    });
}

export function genP20ModelLayout(shape: IModelShape, gptGpuModel: IGptModelLink | null = null, offset: Vec3 = new Vec3(0, 0, 0)) {
    let layout = genGptModelLayout(shape, gptGpuModel, offset);
    applyP20VisualTreatment(layout);
    return layout;
}

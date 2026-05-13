import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    if (state.walkthroughVariant === 'p20') {
        commentary(wt, null, 0)`

And that's a complete P20-style hybrid block!

The shell is intentionally transformer-like: residual stream, layer norm, self-attention, projection,
residual addition, and the final output path are still present. That is why this architecture is close
enough to nanoGPT to compare directly in a controlled ablation.

The difference is the role of the feed-forward side. Instead of only applying an independent MLP to each
token column, P20 uses a rotary gated recurrent state update to carry a compact state through the sequence
and then write a readout back into the residual stream.

So the similarity is high at the block boundary, and lower inside the FFN seam. That is the useful mental
model: P20 is a transformer-shaped LM with a recurrent-control substitute for part of the ordinary MLP
capacity, not a full attention-free architecture in this visualization.
`;
        return;
    }

    commentary(wt, null, 0)`

And that's a complete transformer block!

These form the bulk of any GPT model and are repeated a number of times, with the output of one
block feeding into the next, continuing the residual pathway.

As is common in deep learning, it's hard to say exactly what each of these layers is doing, but we
have some general ideas: the earlier layers tend to focus on learning
lower-level features and patterns, while the later layers learn to recognize and understand
higher-level abstractions and relationships. In the context of natural language processing, the
lower layers might learn grammar, syntax, and simple word associations, while the higher layers
might capture more complex semantic relationships, discourse structures, and context-dependent meaning.

`;

}

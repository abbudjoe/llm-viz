import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, embed, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let minGptLink = 'https://github.com/karpathy/minGPT';
let pytorchLink = 'https://pytorch.org/';
let andrejLink = 'https://karpathy.ai/';
let zeroToHeroLink = 'https://karpathy.ai/zero-to-hero.html';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    setInitialCamera(state, new Vec3(184.744, 0.000, -636.820), new Vec3(296.000, 16.000, 13.500));

    if (state.walkthroughVariant === 'p20') {
        commentary(wt, null, 0)`
Before we delve into the P20-specific seam, let's level-set the comparison.

P20 / RGRP is deliberately close to nanoGPT at the outside: token embeddings, positional information,
layer normalization, causal self-attention, residual additions, final normalization, and the language-model
head all remain transformer-shaped.

This visualization keeps the tiny nanoGPT dimensions so the original chapter-by-chapter animations remain
usable. The current scaling question is happening around a 50M-parameter model with d_model=448; this
browser scene is showing the insertion contract, not drawing every full-scale tensor.

The shared chapters still use the bundled nanoGPT model. The P20 Control chapter now layers in a tiny
browser-side recurrence, so the seam/control/readout animation can show computed toy gates, angles, state,
and residual values. It also includes a scale-faithful inset for the current 50M controller configuration:
packed 448 -> 1,568 controls, four 112-wide state blocks, and 64-padded rotary-pair tiles.

The ablation is narrower than "replace the transformer." It asks whether the feed-forward side of selected
blocks can be made more stateful by using a small rotary gated recurrent update primitive. That primitive
receives the residual vector, creates gates / angles / candidates with a packed projection, updates a compact
state, and writes a readout back into the residual stream.

So as you move through the chapters, most steps are shared with nanoGPT. The chapter named "P20 Control"
is where the new primitive appears.
`;
        return;
    }

    commentary(wt, null, 0)`
Before we delve into the algorithm's intricacies, let's take a brief step back.

This guide focuses on _inference_, not training, and as such is only a small part of the entire machine-learning process.
In our case, the model's weights have been pre-trained, and we use the inference process to generate output. This runs directly in your browser.

The model showcased here is part of the GPT (generative pre-trained transformer) family, which can be described as a "context-based token predictor".
OpenAI introduced this family in 2018, with notable members such as GPT-2, GPT-3, and GPT-3.5 Turbo, the latter being the foundation of the widely-used ChatGPT.
It might also be related to GPT-4, but specific details remain unknown.

This guide was inspired by the ${embedLink('minGPT', minGptLink)} GitHub project, a minimal GPT implementation in ${embedLink('PyTorch', pytorchLink)}
created by ${embedLink('Andrej Karpathy', andrejLink)}.
His YouTube series ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} and the minGPT project have been invaluable resources in the creation of this
guide. The toy model featured here is based on one found within the minGPT project.

Alright, let's get started!
`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.

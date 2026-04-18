import React from 'react';
import s from './FractalP20Panel.module.scss';
import { useProgramState } from '../Sidebar';
import clsx from 'clsx';

const flowSteps = [
    'Prelude transformer blocks convert visible tokens into residual-stream features.',
    'One packed input projection splits each feature into update gate, rotary angle, candidate state, and output gate.',
    'The recurrent state is transformed, rotated by the token-conditioned angle, and gated against the fresh candidate.',
    'The emitted recurrent output controls the looped middle attention scaffold instead of replacing attention outright.',
    'Coda blocks and the LM head turn the refined residual stream into next-token logits.',
];

export const FractalP20Panel: React.FC = () => {
    let progState = useProgramState();
    let isP20Selected = progState.walkthroughVariant === 'p20';

    if (!isP20Selected) {
        return null;
    }

    function togglePanel() {
        progState.showP20Panel = !progState.showP20Panel;
        progState.markDirty();
    }

    return <aside className={clsx(s.panel, !progState.showP20Panel && s.collapsed)} aria-label="Fractal P20 recurrent-control model panel">
        <button className={s.headerButton} onClick={togglePanel} aria-expanded={progState.showP20Panel}>
            <div>
                <p className={s.eyebrow}>Fractal research lane</p>
                <h2 className={s.title}>P20: rotary gated recurrent state update</h2>
                <p className={s.scaleNote}>Toy geometry. Research result is 9.87M params.</p>
            </div>
            <span className={s.toggleHint}>{progState.showP20Panel ? 'Hide explainer' : 'Show explainer'}</span>
        </button>

        {progState.showP20Panel && <div className={s.body}>
            <p className={s.lead}>
                The canvas shows a nanoGPT-scale teaching layout so the original chapter animations
                still work. It is not a to-scale rendering of the 9.87M-parameter Fractal run. The
                important visual claim is architectural placement: transformer attention remains visible,
                while the highlighted middle scaffold swaps vanilla MLP-side work for a rotary gated
                recurrent state update and shared state highway.
            </p>

            <section className={s.section}>
                <h3>Forward path</h3>
                <div className={s.flow}>
                    {flowSteps.map((step) => <div className={s.flowStep} key={step}>
                        <span className={s.dot} />
                        <span>{step}</span>
                    </div>)}
                </div>
            </section>

            <section className={s.section}>
                <h3>Primitive update</h3>
                <pre className={s.formula}>{`(a_t, theta_t, c_t, r_t) = W_in x_t
u_t = rotate(W_s s_{t-1}, theta_t)
s_t = gate(a_t) * u_t + (1 - gate(a_t)) * tanh(c_t)
o_t = gate(r_t) * s_t`}</pre>
            </section>

            <section className={s.section}>
                <h3>Current evidence boundary</h3>
                <div className={s.metrics}>
                    <div className={s.metric}>
                        <strong>9.87M</strong>
                        <span>research-lane model size, not this toy canvas scale</span>
                    </div>
                    <div className={s.metric}>
                        <strong>14L band</strong>
                        <span>roughly matches deeper attention with fewer params</span>
                    </div>
                    <div className={s.metric}>
                        <strong>16L wins</strong>
                        <span>deeper attention still wins on seed-42 loss</span>
                    </div>
                </div>
            </section>

            <p className={s.note}>
                The honest claim is not "attention replacement." The current claim is narrower:
                recurrent control plus looped middle attention looks promising enough to justify a
                30M-50M parameter GPU-grant follow-up.
            </p>
        </div>}
    </aside>;
};

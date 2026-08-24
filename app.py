import streamlit as st
import numpy as np
from rnn_lib import build_vocab, CharModelScratch, train

st.title("✍️ StoryWeaver")
st.caption("Pick RNN or GRU, train it on some text, then generate.")

DEFAULT_TEXT = """the cat sat on the mat
the cat ate the fish
the dog sat on the rug
the dog ate the bone"""

text = st.text_area("Training text", value=DEFAULT_TEXT, height=150)
arch = st.selectbox("Architecture", ["RNN", "GRU"])

if "models" not in st.session_state:
    st.session_state.models = {}
    st.session_state.vocab = None

if st.button("Train", type="primary"):
    if len(text) < 25:
        st.error(f"Training text is only {len(text)} characters — please use at least 25.")
    else:
        char_to_idx, idx_to_char = build_vocab(text)
        idx_seq = [char_to_idx[c] for c in text]
        model = CharModelScratch(len(char_to_idx), arch=arch, n_a=32, seed=1)

        # seq_len can't be longer than the text allows
        seq_len = min(20, len(idx_seq) - 1)

        progress = st.progress(0)
        status = st.empty()
        steps = 400

        def on_step(step, loss):
            if step % 20 == 0:
                progress.progress(min(step / steps, 1.0))
                status.write(f"step {step}, loss {loss:.3f}")

        train(model, idx_seq, steps=steps, seq_len=seq_len, lr=0.1, on_step=on_step)

        st.session_state.models[arch] = model
        st.session_state.vocab = (char_to_idx, idx_to_char)
        st.success(f"Trained {arch}!")

st.divider()

seed = st.text_input("Starting text", value="the cat")
num_chars = st.slider("How many characters to generate", 20, 200, 60, 10)

if st.button("Generate"):
    if arch not in st.session_state.models:
        st.warning(f"Train {arch} first.")
    else:
        model = st.session_state.models[arch]
        char_to_idx, idx_to_char = st.session_state.vocab
        try:
            seed_idx = [char_to_idx[c] for c in seed]
            rng = np.random.default_rng()
            gen = model.generate(seed_idx, num_chars, temperature=0.6, rng=rng)
            st.write(seed + "".join(idx_to_char[i] for i in gen))
        except KeyError as e:
            st.error(f"Character {e} wasn't in the training text.")
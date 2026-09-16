"""Interactive demo: identify (G, K) from experimental displacement data in ~0.1s.

Runs entirely against the pre-trained ROM artifacts in `data/sample/` -- no
FEniCSx/dolfinx required, so this deploys as-is on Streamlit Community Cloud /
HF Spaces. Launch locally with:

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from romid.data import load_experimental_displacements, load_reference_nodes
from romid.inverse import identify_rom
from romid.rom import reconstruct_from_basis
from romid.surrogate import RBFSurrogate
from romid.viz import plot_fields

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

st.set_page_config(page_title="Material Parameter Identification", page_icon="🔩", layout="wide")


@st.cache_resource
def load_rom_artifacts():
    basis = np.loadtxt(DATA_DIR / "reduced_basis.csv", delimiter=",")
    surrogate = RBFSurrogate.load(DATA_DIR / "rbf_surrogate.pkl")
    node_x, node_y = load_reference_nodes(DATA_DIR / "reference_nodes.csv")
    return basis, surrogate, node_x, node_y


st.title("Material Parameter Identification from Full-Field Displacement Data")
st.markdown(
    """
Identify a tensile specimen's **shear modulus (G)** and **bulk modulus (K)** from
Digital Image Correlation (DIC) displacement data, using a POD + RBF **reduced-order
model** trained as a surrogate for a FEniCSx full-order FEM solve.

The ROM surrogate identifies parameters in **~0.1 seconds**, versus ~80 seconds for
the full-order model, at comparable accuracy — see `docs/theory.md` in the repo
for the full derivation and FOM/ROM benchmark.
"""
)

basis, surrogate, node_x, node_y = load_rom_artifacts()

st.sidebar.header("Experimental data")
use_sample = st.sidebar.checkbox("Use the bundled sample dataset", value=True)

uploaded = None
if not use_sample:
    uploaded = st.sidebar.file_uploader(
        "Upload a DIC displacement CSV (columns: x, y, ux, uy, in mm)", type="csv"
    )

experimental_path = DATA_DIR / "experimental_displacements.csv"
if uploaded is not None:
    experimental_path = uploaded

run = st.sidebar.button("Identify material parameters", type="primary")

if run:
    with st.spinner("Interpolating experimental data onto the FEM grid..."):
        exp_ux, exp_uy = load_experimental_displacements(experimental_path, node_x, node_y)

    with st.spinner("Running ROM-based inverse identification..."):
        result = identify_rom(surrogate, basis, exp_ux, exp_uy)

    col1, col2, col3 = st.columns(3)
    col1.metric("Shear modulus G", f"{result.G / 1e9:.2f} GPa")
    col2.metric("Bulk modulus K", f"{result.K / 1e9:.2f} GPa")
    col3.metric("Identification time", f"{result.elapsed_seconds * 1000:.1f} ms")
    st.caption(f"Converged in {result.n_iterations} iterations, {result.n_function_evals} function evaluations.")

    predicted_coeffs = surrogate.predict(result.G, result.K)
    rom_ux, rom_uy = reconstruct_from_basis(basis, predicted_coeffs, len(node_x))

    st.subheader("Experimental vs. ROM-reconstructed displacement (X direction)")
    fig, _ = plot_fields(node_x, node_y, {"Experimental X Displacement": exp_ux, "ROM X Displacement": rom_ux})
    st.pyplot(fig)

    st.subheader("Experimental vs. ROM-reconstructed displacement (Y direction)")
    fig, _ = plot_fields(node_x, node_y, {"Experimental Y Displacement": exp_uy, "ROM Y Displacement": rom_uy})
    st.pyplot(fig)
else:
    st.info("Choose a dataset in the sidebar and click **Identify material parameters** to run the ROM.")

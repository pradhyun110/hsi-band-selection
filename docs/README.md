# Block diagrams

| File | Shows |
|---|---|
| `phase1_methodology.png` | Phase 1 (software): data -> entropy pre-screening -> splits -> 8 band-selection methods -> SVM -> metrics -> decision |
| `fisher_cholesky_flowchart.png` | The chosen algorithm, step by step |
| `phase2_fpga_architecture.png` | Phase 2 (planned): FPGA blocks on the Nexys 4 (Artix-7 XC7A100T) |
| `phase2_workflow.png` | Phase 2 (planned): fixed-point model -> Verilog -> simulation -> synthesis -> board test -> evaluation |

Each `.png` is drawn from the `.dot` file with the same name (Graphviz): `dot -Tpng -Gdpi=170 phase1_methodology.dot -o phase1_methodology.png`.

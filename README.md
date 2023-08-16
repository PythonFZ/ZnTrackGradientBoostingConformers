# Train a Gradient Boosting Regressor on Conformations of EtOH

This Tutorial will provide the code for the following workflow:

```mermaid
flowchart TD

    generate[Generate Structure from SMILES]
    label[Compute Energies using xTB]
    splitDataSet[Split dataset into train & test]
    train[Train a GradientBoostingRegressor]
    evaluate[Evaluate the Model on different metrics]

    generate-->label-->splitDataSet-->train-->evaluate
```

You need to have the `pip install -r requirements.txt` installed.
Additionally, we will use `xtb` which can be installed via conda:

`conda install -c conda-forge xtb-python`
`conda install xtb`

# Fiscal 2025 variant

[← README](../README.md)

`us_hank.fiscal_2025` implements the debt, tax-cut, spending, and repayment
paths published in Auclert, Rognlie, and Straub (2025), *“Fiscal and Monetary
Policy with Heterogeneous Agents.”* It is a clean-room implementation from the
paper: the companion `shade-econ/annual-review` repository currently declares
no license, so none of its notebook/module code or calibration inputs are
copied here.

```python
from us_hank import fiscal_2025

paths = fiscal_2025.fiscal_paths("deficit_tax_cut", size=0.01, T=300)
paths["debt"]
paths["tax_revenue"]
```

The heterogeneous-agent `fiscal_2025.shock()` response is intentionally hard
blocked. Shipping it requires licensed calibration inputs or author permission,
followed by numerical reproduction of the paper's Figure 2. The tested fiscal
paths are infrastructure for that validation; they are not presented as a
validated HANK impulse response.

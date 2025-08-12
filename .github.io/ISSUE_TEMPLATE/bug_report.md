---
name: 🐛 Bug Report
about: Create a report to help us improve MFSU
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ['MiguelAngelFrancoLeon']
---

# 🐛 Bug Report

## 📋 Summary
**Describe the bug in one clear sentence:**

## 🔬 Scientific Context
**Which MFSU domain is affected?**
- [ ] CMB Analysis (Cosmic Microwave Background)
- [ ] Superconductor Critical Temperature Prediction
- [ ] Anomalous Diffusion Modeling
- [ ] Fractal Dimension Calculation
- [ ] Cross-domain Validation
- [ ] Statistical Analysis
- [ ] Visualization/Plotting
- [ ] Docker Environment
- [ ] Other: ___________

**Does this affect the Universal Fractal Constant δF = 0.921?**
- [ ] Yes - δF calculation is incorrect
- [ ] No - δF validation still passes
- [ ] Unknown - needs investigation

## 🔄 Reproduction Steps
**Steps to reproduce the behavior:**

1. Environment setup:
   ```bash
   # Include conda/pip install commands
   
   ```

2. Code that triggers the bug:
   ```python
   # Minimal reproducible example
   import mfsu
   
   ```

3. Expected vs Actual behavior:
   - **Expected:** 
   - **Actual:** 

## 📊 Data and Results
**Include relevant data/outputs:**

```
# Paste error messages, incorrect results, or logs here

```

**Scientific validation check:**
```python
# Run this and paste results:
import mfsu
from mfsu.validation import validate_universal_constant
result = validate_universal_constant()
print(f"δF = {result['delta_F']:.6f} ± {result['uncertainty']:.6f}")
```

## 🖥️ Environment Information
**Complete this information:**

- **MFSU Version:** (run `python -c "import mfsu; print(mfsu.__version__)"`)
- **Python Version:** (run `python --version`)
- **Operating System:** [Windows/macOS/Linux + version]
- **Installation Method:** [pip/conda/Docker/source]
- **GPU Available:** [Yes/No, if relevant]

**Package versions:**
```bash
# Run and paste output:
pip list | grep -E "(numpy|scipy|matplotlib|astropy|healpy)"
```

**Docker information (if using containers):**
```bash
# If using Docker, run and paste:
docker --version
docker-compose --version
```

## 📁 Additional Files
**Attach relevant files if helpful:**
- [ ] Input data files (if small < 1MB)
- [ ] Configuration files
- [ ] Screenshots of plots/errors
- [ ] Log files

## 🧪 Attempted Solutions
**What have you tried to fix this?**

- [ ] Reinstalled MFSU
- [ ] Updated dependencies  
- [ ] Checked documentation
- [ ] Searched existing issues
- [ ] Tried different parameters
- [ ] Other: ___________

## 💡 Expected Scientific Impact
**How does this bug affect scientific results?**

- [ ] Critical - Invalid δF calculation
- [ ] High - Incorrect scientific results
- [ ] Medium - Affects reproducibility
- [ ] Low - Minor convenience issue

**Urgency level:**
- [ ] Critical - Blocks publication/research
- [ ] High - Affects daily work
- [ ] Medium - Workaround available
- [ ] Low - Can wait for next release

## 📋 Additional Context
**Any other context about the problem:**

## 🔍 Self-Check
Before submitting, please confirm:

- [ ] I have checked that this issue hasn't been reported already
- [ ] I have provided a minimal reproducible example
- [ ] I have included all relevant environment information
- [ ] I have tested with the latest MFSU version
- [ ] I have run the δF validation check above

## 📞 Contact Information
**For urgent scientific questions:**
- Email: miguelfranco@mfsu-model.org
- Repository: https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/

---

**Thank you for helping improve MFSU! 🌌**  
*Every bug report helps us make the Universal Fractal Constant δF = 0.921 more accessible to the scientific community.*

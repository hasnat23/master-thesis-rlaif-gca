# Reward-Model Preference Construction — Results Report

**Judge:** `CogComp/bart-faithful-summary-detector`
**Method:** Deterministic sequence-classification model — no OpenAI/GPT calls.

## 1. Holistic Scoring
| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Preference pairs (A wins) | 36 |
| Preference pairs (B wins) | 26 |
| Total usable for DPO | **62** (31.0%) |
| Ties / no_preference | 138 |
| Mean faithfulness score (A) | 0.3053 |
| Mean faithfulness score (B) | 0.3021 |
| Margin threshold | 0.05 |

## 2. Granular (GCA) Scoring
| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Preference pairs (A wins) | 8 |
| Preference pairs (B wins) | 9 |
| Total usable for DPO | **17** (8.5%) |
| Ties / no_preference | 183 |
| Mean GCA score (A) | 0.2823 |
| Mean GCA score (B) | 0.2840 |
| Avg sentences per summary (A / B) | 7.0 / 6.7 |
| Low-faithfulness sentences total (A / B) | 1100 / 1029 |
| Margin threshold | 0.05 |
| GCA alpha | 0.5 |

## 3. Agreement Between Holistic and GCA Decisions
| Metric | Value |
|--------|-------|
| Shared samples | 200 |
| Agree | 149 (74.5%) |
| Disagree | 51 |

### Disagreement Examples

**`cnn_00009_9ca24ec65ab9`**
- Holistic: `B`  (A=0.1610, B=0.2570)
- GCA:      `no_preference`  (A=0.1065, B=0.1143)
- Article snippet: *(CNN) -- Police and FBI agents are investigating the discovery of an empty rocket launcher tube on the front lawn of a Jersey City, New Jersey, home, FBI spokesman Sean Quinn said. Niranjan Desai disc...*
- Summary A: *An empty AT4 anti-tank rocket launcher tube, which is no longer operable, was found on the front lawn of a home in Jersey City, New Jersey. The launch...*
- Summary B: *The news article discusses the discovery of an AT4 anti-tank rocket launcher tube on the lawn of a Jersey City, New Jersey home by a local resident, N...*

**`cnn_00012_be1d204205b5`**
- Holistic: `A`  (A=0.1689, B=0.1113)
- GCA:      `no_preference`  (A=0.1318, B=0.1432)
- Article snippet: *BREMEN, Germany -- Carlos Alberto, who scored in FC Porto's Champions League final victory against Monaco in 2004, has joined Bundesliga club Werder Bremen for a club record fee of  7.8 million euros ...*
- Summary A: *Brazilian midfielder Carlos Alberto has joined Bundesliga club Werder Bremen for a club-record fee of 7.8 million euros ($10.7 million) after a succes...*
- Summary B: *Brazilian footballer Carlos Alberto signed a deal with Bundesliga club Werder Bremen, for a club record fee of 7.8 million euros, after impressing at ...*

**`cnn_00022_b51dd5c0fb1f`**
- Holistic: `A`  (A=0.4375, B=0.3094)
- GCA:      `no_preference`  (A=0.4709, B=0.4393)
- Article snippet: *(CNN) -- A former government contract employee was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk, the Department of Justice announce...*
- Summary A: *Printer-friendly version

A former government contractor, Roy Lynn Oakley, was indicted for stealing restricted nuclear energy-related materials from ...*
- Summary B: *Printers and text messaging Share This Story Copy Link URL

A former government contractor, Roy Lynn Oakley, was indicted on charges of stealing restr...*

**`cnn_00024_799e457f0d3a`**
- Holistic: `A`  (A=0.2386, B=0.1667)
- GCA:      `no_preference`  (A=0.2289, B=0.1951)
- Article snippet: *HONG KONG, China (Reuters) -- Paul Lee got his liver from an executed Chinese prisoner; Karam in Egypt bought a kidney for his sister for $5,300; in Istanbul Hakan is holding out for $30,700 for one o...*
- Summary A: *00,000 rupees ($7,500) for a kidney, Naqvi said. "We have a shortage of donors and we have a shortage of funds ... it's a desperate situation," Naqvi ...*
- Summary B: *00,000 rupees ($6,150) each for a kidney. The trade in Pakistan is controlled by local gangs with the doctors who carry out the transplants colluding ...*

**`cnn_00047_326ca6fe7156`**
- Holistic: `B`  (A=0.0117, B=0.0881)
- GCA:      `no_preference`  (A=0.0221, B=0.0456)
- Article snippet: *(CNN)  -- With his hands and feet shackled and his face obscured by his long hair, Chester Arthur Stiles made his initial court appearance in Las Vegas, Nevada, on Wednesday morning on charges stemmin...*
- Summary A: *A 37-year-old man named Chester Arthur Stiles made his first court appearance in Las Vegas, Nevada, on Wednesday, charged with the videotaped rape of ...*
- Summary B: *Chester Arthur Stiles, a 37-year-old man, appeared in court in Las Vegas, Nevada, on Wednesday for charges related to the videotaped rape of a 2-year-...*


## 4. GCA: Detected Low-Faithfulness Sentences (score < 0.3)
- `cnn_00009_9ca24ec65ab9` | Summary **A** | score=0.1513 | gca_decision=no_preference
  > An empty AT4 anti-tank rocket launcher tube, which is no longer operable, was found on the front lawn of a home in Jersey City, New Jersey.
- `cnn_00009_9ca24ec65ab9` | Summary **A** | score=0.1010 | gca_decision=no_preference
  > The launcher has been turned over to the U.S.
- `cnn_00009_9ca24ec65ab9` | Summary **A** | score=0.1071 | gca_decision=no_preference
  > Army for investigation, and police and the FBI are looking into the origin of the rocket launcher and the circumstances surrounding its appearance on residential property.
- `cnn_00009_9ca24ec65ab9` | Summary **A** | score=0.1040 | gca_decision=no_preference
  > The device, which is a shoulder-fired, direct-fire weapon used against ground targets, is not wire-guided and is not considered a hazard to public safety.
- `cnn_00009_9ca24ec65ab9` | Summary **A** | score=0.0993 | gca_decision=no_preference
  > The nearest military base, Fort Dix, is more than 70 miles from Jersey City.

## 5. Interpretation
- Holistic judging yielded **62** usable DPO pairs out of 200 samples.
- GCA judging yielded **17** usable DPO pairs out of 200 samples.
- The two methods agree on **74.5%** of shared decisions.

Disagreements are expected and are the central interest of the thesis: holistic scoring collapses all sentence evidence into a single number, while GCA can detect a single low-faithfulness sentence that drags the aggregate score below the margin even when the mean score would be similar. These cases demonstrate the value of sentence-level localisation.

---
*Generated by `src/analysis/analyze_reward_preferences.py`*
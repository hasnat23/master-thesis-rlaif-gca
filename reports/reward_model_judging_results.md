# Reward-Model Preference Construction — Results Report

**Judge:** `yzha/AlignScore` (AlignScore-base, RoBERTa backbone)
**Method:** AlignScore NLI-based faithfulness scorer (`nli_sp` mode) — no OpenAI/GPT calls.

## 1. Holistic Scoring
| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Preference pairs (A wins) | 109 |
| Preference pairs (B wins) | 45 |
| Total usable for DPO | **154** (77.0%) |
| Ties / no_preference | 46 |
| Mean faithfulness score (A) | 0.7324 |
| Mean faithfulness score (B) | 0.6530 |
| Margin threshold | 0.05 |

## 2. Granular (GCA) Scoring
| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Preference pairs (A wins) | 97 |
| Preference pairs (B wins) | 60 |
| Total usable for DPO | **157** (78.5%) |
| Ties / no_preference | 43 |
| Mean GCA score (A) | 0.4066 |
| Mean GCA score (B) | 0.3337 |
| Avg sentences per summary (A / B) | 7.0 / 6.7 |
| Low-faithfulness sentences total (A / B) | 372 / 493 |
| Margin threshold | 0.05 |
| GCA alpha | 0.5 |

## 3. Agreement Between Holistic and GCA Decisions
| Metric | Value |
|--------|-------|
| Shared samples | 200 |
| Agree | 121 (60.5%) |
| Disagree | 79 |

### Disagreement Examples

**`cnn_00003_52e70af982e3`**
- Holistic: `no_preference`  (A=0.7245, B=0.7196)
- GCA:      `A`  (A=0.2372, B=0.1212)
- Article snippet: *WASHINGTON (CNN) -- Doctors removed five small polyps from President Bush's colon on Saturday, and "none appeared worrisome," a White House spokesman said. The polyps were removed and sent to the Nati...*
- Summary A: *President Bush underwent a colonoscopy on Saturday at the National Naval Medical Center in Bethesda, Maryland, where doctors removed five small polyps...*
- Summary B: *President Bush underwent a colonoscopy procedure on Saturday to remove five small polyps from his colon at the National Naval Medical Center in Bethes...*

**`cnn_00012_be1d204205b5`**
- Holistic: `A`  (A=0.7951, B=0.7340)
- GCA:      `B`  (A=0.5970, B=0.6643)
- Article snippet: *BREMEN, Germany -- Carlos Alberto, who scored in FC Porto's Champions League final victory against Monaco in 2004, has joined Bundesliga club Werder Bremen for a club record fee of  7.8 million euros ...*
- Summary A: *Brazilian midfielder Carlos Alberto has joined Bundesliga club Werder Bremen for a club-record fee of 7.8 million euros ($10.7 million) after a succes...*
- Summary B: *Brazilian footballer Carlos Alberto signed a deal with Bundesliga club Werder Bremen, for a club record fee of 7.8 million euros, after impressing at ...*

**`cnn_00015_43d38103d1e8`**
- Holistic: `no_preference`  (A=0.6432, B=0.6805)
- GCA:      `B`  (A=0.3122, B=0.5188)
- Article snippet: *WASHINGTON (CNN) -- There is "no remaining hope" of finding six men trapped for almost a month in a Utah coal mine alive, a federal official said Saturday. Isaac Arellano holds a candle and sings duri...*
- Summary A: *The search for six miners trapped in a Utah coal mine since August 6 has been called off after 25 days, with federal officials stating that there is n...*
- Summary B: *It was announced on Saturday, September 8, that the six miners trapped in a Utah coal mine for almost a month are believed to have died as search effo...*

**`cnn_00016_49ab4e627fac`**
- Holistic: `no_preference`  (A=0.9414, B=0.9257)
- GCA:      `A`  (A=0.8257, B=0.7624)
- Article snippet: *(CNN) -- At least 14 people were killed and 60 others wounded Thursday when a bomb ripped through a crowd waiting to see Algeria's president in Batna, east of the capital of Algiers, the Algerie Press...*
- Summary A: *A bomb explosion in Batna, Algeria killed at least 14 people and injured 60 others while they were waiting to see Algeria's president, Abdel-Aziz Bout...*
- Summary B: *A bomb explosion in Batna, Algeria, during a crowd waiting to see President Abdel-Aziz Bouteflika, on Thursday (December 18th, 2008) resulted in the d...*

**`cnn_00022_b51dd5c0fb1f`**
- Holistic: `no_preference`  (A=0.8301, B=0.8555)
- GCA:      `A`  (A=0.3941, B=0.2814)
- Article snippet: *(CNN) -- A former government contract employee was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk, the Department of Justice announce...*
- Summary A: *Printer-friendly version

A former government contractor, Roy Lynn Oakley, was indicted for stealing restricted nuclear energy-related materials from ...*
- Summary B: *Printers and text messaging Share This Story Copy Link URL

A former government contractor, Roy Lynn Oakley, was indicted on charges of stealing restr...*


## 4. GCA: Detected Low-Faithfulness Sentences (score < 0.3)
- `cnn_00000_b0b34b2d843e` | Summary **A** | score=0.2321 | gca_decision=A
  > Despite the newfound wealth, he has stated that he does not plan to be extravagant, and will instead use his money to buy books, CDs, and DVDs.
- `cnn_00000_b0b34b2d843e` | Summary **B** | score=0.0495 | gca_decision=A
  > Instead, he plans to use the money to buy books, CDs, and DVDs.
- `cnn_00003_52e70af982e3` | Summary **A** | score=0.1620 | gca_decision=A
  > He resumed his activities at Camp David after the procedure, speaking to his wife and playing with his dogs.
- `cnn_00003_52e70af982e3` | Summary **A** | score=0.0747 | gca_decision=A
  > Vice President Dick Cheney assumed presidential power during the procedure, but no official action was required before Bush reclaimed power at 9:21 a.m.
- `cnn_00003_52e70af982e3` | Summary **B** | score=0.0269 | gca_decision=A
  > Richard Tubb, and Vice President Dick Cheney assumed presidential power during the two-hour surgery.

## 5. Interpretation
- Holistic judging yielded **154** usable DPO pairs out of 200 samples.
- GCA judging yielded **157** usable DPO pairs out of 200 samples.
- The two methods agree on **60.5%** of shared decisions.

Disagreements are expected and are the central interest of the thesis: holistic scoring collapses all sentence evidence into a single number, while GCA can detect a single low-faithfulness sentence that drags the aggregate score below the margin even when the mean score would be similar. These cases demonstrate the value of sentence-level localisation.

---
*Generated by `src/analysis/analyze_reward_preferences.py`*
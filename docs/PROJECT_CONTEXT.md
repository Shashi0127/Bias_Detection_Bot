Perfect approach! Let me create a comprehensive `PROJECT_CONTEXT.md` file that will give complete context to the new chat:

## **PROJECT_CONTEXT.md**

```markdown
# Universal Bias Detection Bot - Project Context

## Project Overview

AI-powered system that detects bias patterns in ANY CSV file without requiring predefined schemas. Combines statistical analysis, machine learning, and natural language processing to identify discrimination patterns and explain them via Telegram bot.

## Core Innovation

- **Universal Detection**: Works with ANY CSV structure - no predefined schemas needed
- **Layered Analysis**: Statistical + ML + Pattern detection ensures robust results
- **Fallback Safety**: If ML fails, statistical analysis always provides results
- **Plain Language**: LLM converts technical findings to understandable explanations

## Technical Stack

- Python 3.8+
- scikit-learn (ML models)
- pandas/numpy (data processing)
- python-telegram-bot (bot interface)
- Groq API (LLM explanations)
- scipy (statistical tests)

## Architecture Flow
```

ANY CSV File Upload (via Telegram)
↓
[Column Analyzer] - Auto-detects protected attributes & outcomes
↓
[Universal Detector] - Orchestrates parallel analysis
↓
┌────┴────┬────────┐
↓ ↓ ↓
Statistical ML Pattern
Engine Model Finder
↓ ↓ ↓
└────┬────┴────────┘
↓
[Verdict Generator] - Combines findings → BIASED/UNBIASED
↓
[LLM/Groq] - Explains in plain language
↓
[Visualizer] - Creates charts
↓
[Report Builder] - Packages everything
↓
[Telegram Bot] - Delivers to user

````

## Current Implementation Status

### ✅ COMPLETED FILES (8/20):
1. **config/settings.py** - All configuration, thresholds, paths
2. **src/utils/logger.py** - Centralized logging system
3. **src/utils/file_manager.py** - File operations, validation, cleanup
4. **src/core/column_analyzer.py** - Detects protected attributes, outcomes, features
5. **src/analysis/statistical_engine.py** - Chi-square, t-tests, ANOVA, disparate impact
6. **src/training/pattern_library.py** - 10+ bias pattern templates
7. **src/training/data_generator.py** - Generates infinite synthetic datasets
8. **src/training/trainer.py** - Trains ensemble models (GB, RF, LR)

### ⏳ REMAINING FILES (12/20):

#### Core Detection (Priority 1):
9. **src/core/universal_detector.py** - Main orchestrator, runs parallel analysis
10. **src/core/verdict_generator.py** - Final BIASED/UNBIASED decision

#### Analysis (Priority 2):
11. **src/analysis/pattern_finder.py** - Clustering, correlations, hidden patterns
12. **src/analysis/report_builder.py** - Combines all findings into report

#### ML Components (Priority 3):
13. **src/ml/model_handler.py** - Loads models, runs predictions
14. **src/ml/preprocessor.py** - Dynamic preprocessing for any structure

#### LLM Integration (Priority 4):
15. **src/llm/groq_client.py** - Groq API communication
16. **src/llm/prompts.py** - Templates for explanations

#### Utilities (Priority 5):
17. **src/utils/visualizer.py** - Charts and graphs

#### Telegram Bot (Priority 6):
18. **src/telegram/validators.py** - Input validation
19. **src/telegram/bot.py** - Bot handlers
20. **main.py** - Entry point (bot only, training in trainer.py)

## Key Design Patterns

### 1. Column Detection Logic (column_analyzer.py)
- Uses regex patterns from settings.py
- Detects by column names AND values
- Returns: `{'protected_attributes': [], 'outcome_variables': [], 'features': []}`

### 2. Statistical Engine (statistical_engine.py)
- Always returns results even with minimal data
- Implements: disparate impact (80% rule), chi-square, t-tests, ANOVA
- Returns: `{'bias_detected': bool, 'confidence': float, 'overall_bias_score': float}`

### 3. Pattern Library (pattern_library.py)
- 10 bias types: direct, threshold, proxy, intersectional, glass ceiling, etc.
- Each pattern has apply_function for synthetic data
- Used for both training and detection

### 4. Training Pipeline (trainer.py)
- Generates synthetic data with known bias patterns
- Trains ensemble: GradientBoosting, RandomForest, LogisticRegression
- 70% train, 30% test split
- Grid search for hyperparameters

## Expected Functionality

### universal_detector.py should:
```python
def analyze_dataset(df: pd.DataFrame) -> Dict:
    # 1. Run column analysis
    # 2. Parallel execution of:
    #    - statistical_engine.analyze_bias()
    #    - model_handler.predict() if compatible
    #    - pattern_finder.find_patterns()
    # 3. Combine all results with confidence weights
    # 4. Return unified analysis
````

### verdict_generator.py should:

```python
def generate_verdict(analysis_results: Dict) -> Dict:
    # 1. Evaluate all findings
    # 2. Calculate overall bias score
    # 3. Determine confidence level
    # 4. Return: {
    #     'verdict': 'BIASED' or 'UNBIASED',
    #     'confidence': 0.0-1.0,
    #     'severity': 'low/medium/high',
    #     'summary': str
    # }
```

## Coding Standards

1. **NO EMOJIS** in any code or comments
2. **Every .py file** must have `if __name__ == "__main__":` block with tests
3. **Logging**: Use `get_logger()` from utils/logger.py
4. **Error Handling**: Graceful degradation (if ML fails, use statistical)
5. **Type Hints**: Use for all function parameters and returns
6. **Docstrings**: Every function needs proper documentation

## Testing Requirements

Each file's `__main__` block should:

- Create synthetic test data
- Run the main functionality
- Print results
- Handle errors gracefully

## Dependencies (requirements.txt)

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
python-telegram-bot==20.7
groq==0.4.1
python-dotenv==1.0.0
faker==20.1.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
```

## Environment Variables (.env)

```
TELEGRAM_BOT_TOKEN=your_token_here
GROQ_API_KEY=your_key_here
ENVIRONMENT=development
DEBUG=True
```

## Next Implementation Priority

1. **universal_detector.py** - Core orchestrator
2. **verdict_generator.py** - Decision maker
3. **pattern_finder.py** - Advanced pattern detection
4. **report_builder.py** - Report generation
5. Continue with remaining files...

## Expected Output Example

```
Input: ANY CSV file
Output:
- Verdict: "BIASED: Significant disparities detected"
- Confidence: 0.87 (High)
- Affected Groups: "Female applicants 35% less likely to be approved"
- Severity: "High"
- Recommendations: ["Review approval criteria", "Implement blind review"]
- Visualizations: [bias_gauge.png, feature_importance.png]
```

## GitHub Repository Structure

```
Bias_Detection_Bot/
├── config/
│   ├── __init__.py
│   └── settings.py ✅
├── src/
│   ├── core/
│   │   ├── column_analyzer.py ✅
│   │   ├── universal_detector.py ⏳
│   │   └── verdict_generator.py ⏳
│   ├── analysis/
│   │   ├── statistical_engine.py ✅
│   │   ├── pattern_finder.py ⏳
│   │   └── report_builder.py ⏳
│   ├── ml/
│   │   ├── model_handler.py ⏳
│   │   └── preprocessor.py ⏳
│   ├── training/
│   │   ├── pattern_library.py ✅
│   │   ├── data_generator.py ✅
│   │   └── trainer.py ✅
│   ├── llm/
│   │   ├── groq_client.py ⏳
│   │   └── prompts.py ⏳
│   ├── telegram/
│   │   ├── bot.py ⏳
│   │   └── validators.py ⏳
│   └── utils/
│       ├── logger.py ✅
│       ├── file_manager.py ✅
│       └── visualizer.py ⏳
├── data/
├── logs/
├── tests/
├── main.py ⏳
├── requirements.txt ✅
├── .env.example ✅
└── README.md ✅
```

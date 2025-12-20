# Translation Annotator

A web-based tool for editing, annotating, and scoring translated conversations. Created by Chengheng Li.

---

## ⚠️ IMPORTANT WARNING

**DO NOT refresh (F5) or close the browser before committing your changes!**

All work done between commits will be **permanently lost** if you refresh or close the page. Always click **"✓ Commit Changes"** to save your progress.

---

## 📋 Requirements

### CSV File Format

Your CSV file must contain these columns:
- `original_conversation` - The original conversation text
- `translated_conversation` - The translated conversation text

Optional columns (will be added if not present):
- `semantic_accuracy_1to5`
- `completeness_1to5`
- `constraints_preserved_yes_no_na`
- `overall_quality_1to5`
- `issues_if_any`

### Conversation Format

Conversations should be formatted with `[USER]:` and `[ASSISTANT]:` prefixes:

```
[USER]: Hello, how are you?

[ASSISTANT]: I'm doing well, thank you for asking!

[USER]: Great to hear!
```

---

## 🚀 Getting Started

### Step 1: Open the Tool

Open `translation_editor_pro.html` in your web browser (Chrome, Firefox, Safari, or Edge recommended).

### Step 2: Upload Your CSV

1. Drag and drop your CSV file onto the drop zone, OR
2. Click "Open CSV" button to browse for your file

### Step 3: Load Previous Work (Optional)

When you upload a CSV, a popup will ask if you want to load a commit history file:
- Click **OK** to select your `{filename}_commits.json` file
- Click **Cancel** to start fresh

> **Note:** Your commit history file is named `{csvname}_commits.json` (e.g., `spanish_commits.json`)

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` | Previous turn |
| `↓` | Next turn |
| `←` | Previous entry |
| `→` | Next entry |
| `Enter` | Edit selected turn |
| `Esc` | Finish editing |

---

## 📝 Editing Translations

### Turn Editor Tab

1. **Select an entry** from the sidebar on the left
2. **Navigate turns** using `↑` and `↓` arrow keys
3. **Edit a turn** by pressing `Enter` or clicking on it
4. **Make your changes** in the text area
5. **Exit edit mode** by pressing `Esc`

### Full Text Tab

View and edit the complete translated conversation as a single text block.

### Show Diff Tab

Compare the original and translated conversations side by side with highlighted differences.

---

## ⭐ Annotating Quality Scores

Go to the **Annotate** tab to rate each entry:

### Scoring Fields

| Field | Values | Description |
|-------|--------|-------------|
| **Semantic Accuracy** | 1-5 | How accurately the meaning is preserved (1=Poor, 5=Excellent) |
| **Completeness** | 1-5 | How complete the translation is (1=Incomplete, 5=Complete) |
| **Constraints Preserved** | Yes/No/N/A | Whether formatting and constraints are maintained |
| **Overall Quality** | 1-5 | Overall translation quality (1=Poor, 5=Excellent) |
| **Issues** | Free text | Describe any issues found |

### Saving Scores

- Scores are automatically tracked when you click the buttons
- Click **"💾 Save Scores"** to confirm
- Scores are included in your commits and exported CSV

---

## 💾 Saving Your Work

### Committing Changes

1. Click **"✓ Commit Changes"** button
2. Enter a commit message (required)
3. Optionally add details
4. Click **"Commit"**

When you commit:
- A `{filename}_commits.json` file is automatically downloaded
- This file contains all your changes and scores
- **Keep this file safe** - you'll need it to resume later!

### Export Options

Click **"📦 Export All"** to access:
- **Corrected CSV** - Your edited translations with scores
- **Changes JSON** - Detailed log of all changes
- **Commit History** - Full commit history
- **Detailed Report** - Comprehensive report of all work

---

## 🔄 Resuming Work

To continue working on a previously edited file:

1. Open the Translation Annotator
2. Upload your CSV file
3. When prompted, click **OK**
4. Select your `{filename}_commits.json` file
5. All your previous changes and scores will be restored!

---

## 📊 Interface Overview

### Header Bar
- **Open CSV** - Load a new CSV file
- **✓ Commit Changes** - Save your work (appears after loading CSV)
- **📦 Export All** - Export options (appears after loading CSV)
- **📥 Import History** - Manually import a commit history file
- **🕐 History** - View commit history

### Sidebar
- Lists all entries in your CSV
- Shows modification status for each entry
- Click to navigate between entries

### Main Content Area
- **Stats Bar** - Shows total entries, modified count, changes, and commits
- **Tabs** - Switch between Turn Editor, Full Text, Show Diff, and Annotate views
- **Floating Toolbar** - Quick navigation and progress indicator

---

## 📁 File Outputs

| File | Description |
|------|-------------|
| `{name}_corrected.csv` | Edited CSV with all changes and scores |
| `{name}_commits.json` | Commit history for resuming work |
| `{name}_changes.json` | Detailed change log |
| `{name}_report.txt` | Human-readable report |

### Corrected CSV Columns

The exported corrected CSV includes:

| Column | Description |
|--------|-------------|
| `original_conversation` | The original source conversation |
| `translated_conversation` | The **original** translated conversation (unchanged) |
| `translated_conversation_corrected` | Your **corrected/edited** translation |
| `semantic_accuracy_1to5` | Your semantic accuracy score |
| `completeness_1to5` | Your completeness score |
| `constraints_preserved_yes_no_na` | Constraints preserved (Yes/No/N/A) |
| `overall_quality_1to5` | Your overall quality score |
| `issues_if_any` | Any issues you documented |

---

## 💡 Tips

1. **Commit frequently** - Don't lose your work!
2. **Use keyboard shortcuts** - Much faster than clicking
3. **Keep your `_commits.json` file** - Essential for resuming work
4. **Check the diff view** - Helps spot translation issues
5. **Use the Issues field** - Document problems for later review

---

## 🐛 Troubleshooting

### "No uncommitted changes to commit"
You haven't made any changes since your last commit. Edit some translations or scores first.

### Lost my work after refresh
Unfortunately, if you didn't commit before refreshing, the work is lost. Always commit before closing!

### Can't load commit history
Make sure you're selecting the correct `{filename}_commits.json` file that matches your CSV.

### Scores not showing in exported CSV
Make sure to click "💾 Save Scores" in the Annotate tab before exporting.

---

## 📞 Support

For issues or questions, contact Chengheng Li.

---

*Translation Annotator v1.0*

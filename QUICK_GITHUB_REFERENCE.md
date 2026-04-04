# 🚀 QUICK REFERENCE - GITHUB & DEVELOPMENT GUIDE

## 📍 YOUR REPOSITORY
```
https://github.com/IRONCODER29122004/identix-deploy
```

---

## 🔄 DAILY COMMANDS

### **Start Working:**
```powershell
cd "d:\link2\Capstone 4-1\Code_try_1"
code .
git pull origin v1.1.0-development
```

### **Make Changes & Save:**
```powershell
# After editing files
git add .
git commit -m "Description of what you changed"
git push origin v1.1.0-development
```

### **Check Status:**
```powershell
git status              # What changed
git log --oneline -5    # Last 5 commits
git diff                # Detailed changes
```

---

## 🌿 SWITCH BRANCHES

```powershell
# See all branches
git branch -a

# Switch to stable version
git checkout main

# Switch to development
git checkout v1.1.0-development

# Create new branch
git checkout -b feature/my-feature
```

---

## 🔐 EMERGENCY ROLLBACK

### **If updates break everything:**
```powershell
# Go back to v1.0.0 instantly
git checkout main
git reset --hard v1.0.0
```

### **If you want to keep most changes:**
```powershell
# Undo last commit but keep files modified
git reset HEAD~1

# Undo last 3 commits
git reset HEAD~3
```

---

## 📊 STORAGE LIMITS (FREE PLAN)

| Thing | Limit | Your Usage | Cost |
|-------|-------|-----------|------|
| Repository Size | ∞ Unlimited | 756 MB | **Free** |
| Single File | 100 MB (LFS OK) | Using LFS | **Free** |
| LFS Storage | 1 GB/month | ~500 MB | **Free** |
| Collaborators | ∞ Unlimited | You only | **Free** |

✅ **You will NOT exceed any limit!**

---

## 🔑 DO YOU NEED A PAT?

| Situation | Need PAT? |
|-----------|-----------|
| Push from VS Code | ❌ NO |
| Push from PowerShell | ❌ NO |
| GitHub Actions/CI | ✅ YES |
| Two-factor auth | ✅ YES |
| Automation scripts | ✅ YES |

**Your current setup:** ❌ NO PAT NEEDED

---

## 📤 PUSH TO GITHUB

```powershell
# Push current branch
git push

# Push to specific branch
git push origin v1.1.0-development

# Push all branches
git push origin --all

# Push tags
git push origin --tags

# Force push (only if needed!)
git push -f origin main
```

---

## 💾 MERGE BRANCHES (When feature is ready)

```powershell
# Go to main
git checkout main

# Pull latest from main
git pull origin main

# Merge feature branch
git merge v1.1.0-development

# Push merged changes
git push origin main

# Tag new version
git tag -a v1.1.0 -m "Added VERITAS AI"
git push origin v1.1.0
```

---

## 🏷️ CREATE VERSION TAGS

```powershell
# Create tag
git tag -a v1.1.0 -m "Description of release"

# Push tag to GitHub
git push origin v1.1.0

# View all tags
git tag -l
```

---

## 🔥 COMMON ISSUES & FIXES

### **Push Failed:**
```powershell
git fetch origin
git pull origin main
git push origin v1.1.0-development
```

### **Merge Conflict:**
```powershell
# Use your version
git checkout --ours .

# Or use their version
git checkout --theirs .

# Then commit
git add .
git commit -m "Resolve conflict"
```

### **Want to Undo Last Commit:**
```powershell
# Keep files (undo commit only)
git reset HEAD~1

# Or delete everything
git reset --hard HEAD~1
```

### **Accidentally on Wrong Branch:**
```powershell
# Save your work temporarily
git stash

# Switch to correct branch
git checkout v1.1.0-development

# Get your work back
git stash pop
```

---

## 📱 CLONE REPO ON ANOTHER COMPUTER

```powershell
git clone https://github.com/IRONCODER29122004/identix-deploy.git
cd identix-deploy
code .
```

---

## 🤝 ADD FRIEND TO REPOSITORY

1. Go to: https://github.com/IRONCODER29122004/identix-deploy
2. Click **Settings**
3. Click **Collaborators** (left menu)
4. Click **Add people**
5. Enter friend's GitHub username
6. Click **Add**

Friend can now:
```powershell
git clone https://github.com/IRONCODER29122004/identix-deploy.git
```

---

## 📊 VIEW ON GITHUB

| What to View | Where | Link |
|-------------|-------|------|
| Files & folders | Code tab | `/` |
| Branches | Dropdown (top-left) | `/branches` |
| Commits | Commits button | `/commits` |
| Releases/Tags | Releases (right) | `/releases` |
| History of file | Click file, use History | `/blame/main/file` |
| Settings | Settings tab | `/settings` |

---

## ✅ VERIFICATION CHECKLIST

- [ ] Repository created on GitHub
- [ ] All files visible on GitHub
- [ ] All branches synced
- [ ] v1.0.0 tag created
- [ ] Can clone repository
- [ ] LFS storage shows models
- [ ] Rollback tested (optional)
- [ ] Friend can access (optional)

---

## 🆘 NEED HELP?

### **Common Questions:**

**Q: Where's my code?**
A: https://github.com/IRONCODER29122004/identix-deploy → Code tab

**Q: How do I see changes?**
A: Click "Commits" button to see all changes with dates

**Q: Can people hack my code?**
A: No! Repository is PRIVATE (only you can see)

**Q: Will I be charged?**
A: No! Everything is FREE (well within limits)

**Q: How do I get my code on another computer?**
A: `git clone https://github.com/IRONCODER29122004/identix-deploy.git`

**Q: How do I backup?**
A: Push regularly! `git push origin v1.1.0-development`

**Q: How do I share with friend?**
A: Invite them as collaborator or send GitHub link

---

## 🎯 DEVELOPMENT WORKFLOW

### **Day 1: Setup**
1. ✅ Created GitHub repository
2. ✅ Pushed all code
3. ✅ Verified on GitHub

### **Day 2+: Development**
```
Morning:
  → git pull (latest changes)
  → code .  (open VS Code)

During Day:
  → Make changes
  → Test features
  → Commit frequently

End of Day:
  → git push (backup to GitHub)
  → git log (verify commits)
```

### **Feature Complete:**
```
→ Test thoroughly
→ Fix any bugs
→ git merge to main
→ git tag new version
→ git push
→ Celebrate! 🎉
```

---

## 📚 LEARN MORE

GitHub Docs: https://docs.github.com
Git Cheat Sheet: https://github.com/joshnh/Git-Commands
Git Tutorial: https://git-scm.com/book/en/v2

---

## ✨ YOU'RE ALL SET!

Everything is ready. Start coding! 🚀

**Key Points:**
- ✅ Code is backed up on GitHub
- ✅ Version control is working
- ✅ Rollback anytime
- ✅ Share with friends
- ✅ All FREE
- ✅ No limits exceeded

Go build VERITAS AI! 🧠💪

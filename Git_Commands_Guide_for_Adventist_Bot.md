# Git Commands Guide for Adventist Bot Project

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding the Adventist Bot Project](#understanding-the-adventist-bot-project)
3. [Initial Setup](#initial-setup)
4. [.gitignore Explained](#gitignore-explained)
5. [Daily Workflow](#daily-workflow)
6. [Branch Management](#branch-management)
7. [Collaboration](#collaboration)
8. [Keeping Organized](#keeping-organized)
9. [Advanced Commands](#advanced-commands)
10. [Troubleshooting](#troubleshooting)
11. [Additional Resources](#additional-resources)

---

## Introduction
Brief overview of Git and GitHub and the importance of version control.

---

## Understanding the Adventist Bot Project
Overview of the file structure (`README.md`, `.gitignore`, `persona`, `.env`, etc.) and explanation of what each file and folder is for.

---

## Initial Setup
### Installing Git
- **Windows**: Download and install from [Git for Windows](https://gitforwindows.org/)
- **MacOS**: Use `brew install git` or download from [Git website](https://git-scm.com/)
- **Linux**: Use package manager, e.g., `sudo apt-get install git`

### Configuring Git
- Set username: `git config --global user.name "Your Name"`
- Set email: `git config --global user.email "your.email@example.com"`

### Cloning the Adventist Bot Repository
- `git clone https://github.com/realgermosen/Adventist_bot.git`

---

## .gitignore Explained
What `.gitignore` is and why it's important. What types of files to include (e.g., `.env`), and how to modify `.gitignore` if needed.

---

## Daily Workflow
- Check status: `git status`
- Add changes: `git add .` or `git add <specific-file>`
- Commit changes: `git commit -m "Your commit message"`
- Push changes: `git push origin <branch-name>`
- Pull updates: `git pull origin <branch-name>`
- View history: `git log`

---

## Branch Management
- Create branch: `git checkout -b <branch-name>`
- Switch branch: `git checkout <branch-name>`
- Merge branches: `git merge <branch-name>`
- Delete branch: `git branch -d <branch-name>`

---

## Collaboration
- Fork repository: Go to GitHub project and click "Fork"
- Create Pull Request: Go to GitHub project -> "Pull Requests" -> "New Pull Request"
- Resolve conflicts: Manually edit files to resolve conflicts, then commit
- Collaborator roles: GitHub settings -> "Collaborators" -> Add by username or email

---

## Keeping Organized
- Stash changes: `git stash`
- Tag releases: `git tag -a v1.0 -m "version 1.0"`
- Update `requirements.txt`: Use `pip freeze > requirements.txt`

---

## Advanced Commands
- Cherry-pick commits: `git cherry-pick <commit-hash>`
- Rewrite history (rebase): `git rebase -i <base-commit>`
- Reset changes: `git reset --hard <commit-hash>`

---

## Troubleshooting
Common errors and how to resolve them.

---

## Additional Resources
- [Official Git Documentation](https://git-scm.com/docs)
- [GitHub Help](https://help.github.com/)

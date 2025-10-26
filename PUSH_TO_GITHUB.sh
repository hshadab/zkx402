#!/bin/bash
# Quick GitHub Push Script
#
# This script helps you push the zkx402 repository to GitHub
# Run this after creating a repository on GitHub

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║   ZKx402 GitHub Push Helper                                  ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}⚠️  Git not initialized. Initializing now...${NC}"
    git init
    git add .
    git commit -m "Initial commit: ZKx402 Fair-Pricing for x402 Protocol"
fi

# Check current status
echo -e "${GREEN}✓ Git repository initialized${NC}"
echo ""
git log --oneline -1
echo ""

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${YELLOW}⚠️  No username provided. Exiting.${NC}"
    exit 1
fi

# Construct repository URL
REPO_URL="https://github.com/${GITHUB_USERNAME}/zkx402.git"

echo ""
echo -e "${BLUE}Repository URL: ${REPO_URL}${NC}"
echo ""

# Ask for confirmation
read -p "Have you created the repository on GitHub? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Please create the repository first:${NC}"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: zkx402"
    echo "3. Description: Zero-Knowledge Fair-Pricing for x402 Protocol"
    echo "4. DO NOT initialize with README (we have one)"
    echo "5. Click 'Create repository'"
    echo ""
    echo "Then run this script again."
    exit 0
fi

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists. Removing...${NC}"
    git remote remove origin
fi

# Add remote
echo -e "${BLUE}Adding remote...${NC}"
git remote add origin "$REPO_URL"

# Rename branch to main
echo -e "${BLUE}Renaming branch to main...${NC}"
git branch -M main

# Push to GitHub
echo ""
echo -e "${BLUE}Pushing to GitHub...${NC}"
echo ""

git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║   ✅ Successfully pushed to GitHub!                          ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Your repository:${NC}"
    echo "https://github.com/${GITHUB_USERNAME}/zkx402"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Add topics for discoverability (Settings → Topics)"
    echo "   - zero-knowledge, zkp, x402, payment-protocol, fair-pricing"
    echo "2. Enable GitHub Pages for documentation (Settings → Pages)"
    echo "3. Add GitHub Actions for CI/CD (see GITHUB_SETUP.md)"
    echo ""
else
    echo ""
    echo -e "${YELLOW}⚠️  Push failed. Common issues:${NC}"
    echo ""
    echo "1. Repository doesn't exist on GitHub"
    echo "   → Create it at https://github.com/new"
    echo ""
    echo "2. Authentication failed"
    echo "   → Run: gh auth login"
    echo "   → Or set up SSH keys: https://docs.github.com/en/authentication"
    echo ""
    echo "3. Remote already has commits"
    echo "   → Use: git push -u origin main --force (⚠️  overwrites remote)"
    echo ""
fi

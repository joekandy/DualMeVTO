#!/bin/bash

# DualMe Virtual Try-On - GitHub Repository Initialization Script

set -e

echo "🚀 Initializing DualMe Virtual Try-On GitHub Repository"
echo "=" * 60

# Configuration
REPO_NAME="dualme-virtual-tryon"
REPO_DESCRIPTION="Advanced AI-powered virtual clothing try-on system ready for deployment on salad.com"
GITHUB_USERNAME=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is installed
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    print_success "Git is installed"
}

# Check if GitHub CLI is installed
check_gh() {
    if ! command -v gh &> /dev/null; then
        print_warning "GitHub CLI is not installed. You'll need to create the repository manually."
        return 1
    fi
    print_success "GitHub CLI is installed"
    return 0
}

# Get GitHub username
get_github_username() {
    if command -v gh &> /dev/null; then
        GITHUB_USERNAME=$(gh api user --jq .login 2>/dev/null || echo "")
    fi
    
    if [ -z "$GITHUB_USERNAME" ]; then
        read -p "Enter your GitHub username: " GITHUB_USERNAME
    fi
    
    if [ -z "$GITHUB_USERNAME" ]; then
        print_error "GitHub username is required"
        exit 1
    fi
    
    print_success "GitHub username: $GITHUB_USERNAME"
}

# Initialize git repository
init_git_repo() {
    print_status "Initializing Git repository..."
    
    if [ ! -d ".git" ]; then
        git init
        print_success "Git repository initialized"
    else
        print_warning "Git repository already exists"
    fi
    
    # Set up .gitattributes for LFS
    cat > .gitattributes << EOF
# Git LFS tracking for large model files
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
EOF
    
    print_success ".gitattributes configured for Git LFS"
}

# Create GitHub repository
create_github_repo() {
    print_status "Creating GitHub repository..."
    
    if command -v gh &> /dev/null; then
        if gh repo view "$GITHUB_USERNAME/$REPO_NAME" &> /dev/null; then
            print_warning "Repository $GITHUB_USERNAME/$REPO_NAME already exists"
            return 0
        fi
        
        gh repo create "$REPO_NAME" \
            --description "$REPO_DESCRIPTION" \
            --public \
            --add-readme=false \
            --clone=false
            
        print_success "GitHub repository created: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    else
        print_warning "GitHub CLI not available. Please create the repository manually at:"
        print_warning "https://github.com/new"
        print_warning "Repository name: $REPO_NAME"
        print_warning "Description: $REPO_DESCRIPTION"
        read -p "Press Enter after creating the repository..."
    fi
}

# Add remote and push
setup_remote() {
    print_status "Setting up remote repository..."
    
    # Add remote
    if git remote get-url origin &> /dev/null; then
        print_warning "Origin remote already exists"
        git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    else
        git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    fi
    
    print_success "Remote origin configured"
}

# Stage and commit files
commit_files() {
    print_status "Staging and committing files..."
    
    # Stage all files
    git add .
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        print_warning "No changes to commit"
        return 0
    fi
    
    # Commit with descriptive message
    git commit -m "feat: initial commit - DualMe Virtual Try-On system

- Advanced AI-powered virtual clothing try-on
- Leffa model integration with GPU optimization
- Gradio web interface with REST API
- Docker containerization for salad.com deployment
- Complete documentation and deployment scripts
- Production-ready with health checks and monitoring

Features:
✨ State-of-the-art virtual try-on technology
🔥 CUDA/GPU optimization for performance
🌐 Beautiful web interface with real-time preview
📡 REST API for integration
☁️ Cloud deployment ready (salad.com optimized)
🛡️ Production hardened with monitoring

Ready for deployment! 🚀"

    print_success "Files committed successfully"
}

# Push to GitHub
push_to_github() {
    print_status "Pushing to GitHub..."
    
    # Set upstream and push
    git branch -M main
    git push -u origin main
    
    print_success "Code pushed to GitHub successfully!"
}

# Create initial release
create_release() {
    if command -v gh &> /dev/null; then
        print_status "Creating initial release..."
        
        gh release create "v1.0.0" \
            --title "🎉 DualMe Virtual Try-On v1.0.0" \
            --notes "🚀 **First Release - Production Ready!**

## ✨ Features

- **Advanced Virtual Try-On**: Powered by state-of-the-art Leffa model
- **GPU Optimized**: Full CUDA support with multi-GPU scaling
- **Web Interface**: Beautiful Gradio UI with real-time preview
- **REST API**: Complete API endpoints for integration
- **Production Ready**: Health checks, monitoring, auto-scaling
- **Cloud Deployable**: Optimized for salad.com deployment

## 🚀 Quick Start

\`\`\`bash
# Clone and setup
git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
cd $REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Run locally
python app_gradio.py
\`\`\`

## 🌊 Deploy on Salad.com

\`\`\`bash
# Automated deployment
python salad_deploy.py
\`\`\`

Ready for production deployment! 🎯" \
            --draft=false \
            --prerelease=false

        print_success "Release v1.0.0 created successfully!"
    else
        print_warning "GitHub CLI not available. Create release manually at:"
        print_warning "https://github.com/$GITHUB_USERNAME/$REPO_NAME/releases/new"
    fi
}

# Generate deployment summary
generate_summary() {
    print_success "Repository setup complete!"
    echo
    echo "📋 Summary:"
    echo "  Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo "  Clone URL: git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo
    echo "🚀 Next Steps:"
    echo "  1. Share the repository with your team"
    echo "  2. Set up CI/CD pipelines (optional)"
    echo "  3. Deploy to salad.com using: python salad_deploy.py"
    echo "  4. Star the repository to show support! ⭐"
    echo
    echo "💡 Quick Commands:"
    echo "  • Local testing: docker-compose up --build"
    echo "  • Model download: python download_models.py"
    echo "  • Salad deployment: python salad_deploy.py"
    echo
    print_success "DualMe Virtual Try-On is ready for the world! 🌍"
}

# Main execution
main() {
    check_git
    
    if check_gh; then
        print_success "Using GitHub CLI for automated setup"
    else
        print_warning "Manual GitHub setup required"
    fi
    
    get_github_username
    init_git_repo
    create_github_repo
    setup_remote
    commit_files
    push_to_github
    create_release
    generate_summary
}

# Run main function
main

echo
echo "🎉 GitHub repository initialization complete!"
echo "   Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "   Ready for deployment on salad.com! 🚀" 
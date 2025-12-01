#!/bin/bash
# Setup script to create .env files for Hugging Face integration

echo "🔧 Setting up environment files for Hugging Face..."

# Create .env.example
cat > .env.example << 'EOF'
# Hugging Face Configuration
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here
HF_USERNAME=your_huggingface_username_here
EOF

echo "✅ Created .env.example"

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env already exists. Skipping creation."
    echo "   Please edit .env manually to add your HF_TOKEN"
else
    # Create .env with instructions
    cat > .env << 'EOF'
# Hugging Face Configuration
# IMPORTANT: Replace 'your_token_here' with your actual Hugging Face token
# To get a token:
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Click "New token"
# 3. Give it a name (e.g., "splade-pt-br-upload")
# 4. Select "write" permission
# 5. Copy the token and paste it below

HF_TOKEN=your_token_here
HF_USERNAME=your_huggingface_username_here
EOF
    echo "✅ Created .env"
fi

echo ""
echo "📝 Next steps:"
echo "1. Get your Hugging Face token from: https://huggingface.co/settings/tokens"
echo "2. Edit .env and replace 'your_token_here' with your actual token"
echo "3. Run: source .env"
echo ""


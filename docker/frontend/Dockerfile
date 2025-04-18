# Use Node.js LTS
FROM node:18-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY src/frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend code
COPY src/frontend ./

# Build the application
RUN npm run build

# Expose frontend port
EXPOSE 3000

# Run the frontend
CMD ["npm", "start"] 
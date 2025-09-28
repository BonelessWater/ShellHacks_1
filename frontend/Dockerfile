# frontend/Dockerfile - Production with Node.js + Express
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Install dependencies first for better caching
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the React app
RUN npm run build

# Expose port 3000 for Express server
EXPOSE 3000

# Start the Express server serving the React build
CMD ["npm", "run", "start:server"]
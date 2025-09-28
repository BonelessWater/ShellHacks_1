# OPTIMIZED Frontend Dockerfile
FROM node:18-alpine AS dependencies

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

WORKDIR /app

# Copy only package files first for better layer caching
COPY package*.json ./

# Install dependencies with optimizations
RUN npm ci --only=production --silent --no-audit --no-fund \
    && npm cache clean --force

# =============================================
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy dependencies from previous stage
COPY --from=dependencies /app/node_modules ./node_modules
COPY package*.json ./

# Install dev dependencies for build
RUN npm ci --silent --no-audit --no-fund

# Copy source code (order matters for caching)
COPY public/ ./public/
COPY src/ ./src/
COPY *.js *.json *.ts *.tsx ./

# Build with optimizations
ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=2048"
ENV GENERATE_SOURCEMAP=false
ENV CI=true

RUN npm run build \
    && rm -rf node_modules \
    && rm -rf src

# =============================================
# Production stage
FROM nginx:alpine AS production

# Install curl for health checks
RUN apk add --no-cache curl dumb-init

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Create nginx cache directory
RUN mkdir -p /var/cache/nginx && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /usr/share/nginx/html

# Use non-root user
USER nginx

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["nginx", "-g", "daemon off;"]
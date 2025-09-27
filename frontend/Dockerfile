# --- Build stage ---
FROM node:18-alpine AS build
WORKDIR /app

COPY package*.json ./
COPY craco.config.js ./craco.config.js

RUN npm ci || npm install
COPY . .

ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build

# --- Nginx stage unchanged ---
FROM nginx:alpine
RUN apk add --no-cache curl
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/build /usr/share/nginx/html
RUN mkdir -p /var/cache/nginx
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1
CMD ["nginx", " -g", "daemon off;"]

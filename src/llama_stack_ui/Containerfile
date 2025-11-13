FROM node:22.5.1-alpine

ENV NODE_ENV=production

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Install llama-stack-ui from npm
RUN npm install -g llama-stack-ui

USER nextjs

ENTRYPOINT ["dumb-init", "--"]
CMD ["llama-stack-ui"]

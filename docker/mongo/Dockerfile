ARG MONGO_VERSION=7.0

FROM mongo:${MONGO_VERSION}

COPY ./scripts/init.js /docker-entrypoint-initdb.d

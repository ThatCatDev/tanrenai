FROM golang:1.26-alpine AS build
RUN apk add --no-cache git
WORKDIR /src
COPY platform/ platform/
RUN cd platform && CGO_ENABLED=0 go build -o /tanrenai-platform .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=build /tanrenai-platform /usr/local/bin/tanrenai-platform
EXPOSE 3000
ENTRYPOINT ["tanrenai-platform"]
CMD ["serve"]

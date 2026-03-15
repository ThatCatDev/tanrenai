FROM golang:1.25-alpine AS build
RUN apk add --no-cache git
WORKDIR /src
COPY server/ server/
RUN cd server && CGO_ENABLED=0 go build -o /tanrenai-server .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=build /tanrenai-server /usr/local/bin/tanrenai-server
EXPOSE 8080
ENTRYPOINT ["tanrenai-server"]
CMD ["serve", "--host", "0.0.0.0"]

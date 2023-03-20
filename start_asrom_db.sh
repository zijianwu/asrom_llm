#!/bin/bash

set -eo pipefail

# Define container name and password
CONTAINER_NAME="asrom_db"
PASSWORD="password"

# Check if container is already running
if [ "$(docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}")" == "$CONTAINER_NAME" ]; then
    echo "Postgres database is already running."
else
    # Check if container exists
    if [ "$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}")" == "$CONTAINER_NAME" ]; then
        # Start existing container
        docker start $CONTAINER_NAME
    else
        # Create new container
        docker run --name $CONTAINER_NAME -e POSTGRES_PASSWORD=$PASSWORD -d -p 5432:5432 postgres
    fi

    echo "Postgres database has been started."
fi

# Get the container's IP address
CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $CONTAINER_NAME)

# Print the port and IP address
echo "You can connect to the Postgres database at:"
echo "  Database name: postgres"
echo "  Host: localhost"
echo "  Port: 5432"

while true; do
    read -p "Enter 'exit' to stop the container: " user_input
    if [[ "$user_input" == "exit" ]]; then
        docker stop $CONTAINER_NAME
        echo "Postgres database has been stopped."
        exit 0
    fi
done

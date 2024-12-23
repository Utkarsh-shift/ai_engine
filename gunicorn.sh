#!/bin/bash

# Restart Gunicorn
echo "Restarting Gunicorn service..."

# Reload Systemd Daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Restart Gunicorn Socket and Service
echo "Restarting Gunicorn socket and service..."

sudo systemctl restart gunicorn
sudo systemctl restart gunicorn.socket 
sudo systemctl restart  gunicorn.service

# Reload Nginx
echo "Reloading Nginx..."
sudo systemctl reload nginx

# Check the Status of Gunicorn
echo "Checking Gunicorn service status..."
sudo systemctl status gunicorn.socket

echo "Checking Gunicorn socket status..."
sudo systemctl status gunicorn.service

# Check the Status of Nginx
echo "Checking Nginx status..."
sudo systemctl status nginx  

sudo systemctl daemon-reload

sudo systemctl start celery

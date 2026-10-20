@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train.ps1" %*

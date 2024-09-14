#!/usr/bin/env bash
# Prevent hard coding bash location

# -e exit on fail
# -o pipefail looks for exit code on pipe command
# -u unset vars cause errors
# -x prints each command
set -euxo pipefail

# ANSI Color codes
readonly RED="\e[31m"
readonly GREEN="\e[32m"
readonly BLUE="\e[34m"
readonly RESET="\e[0m"
readonly BOLD="\e[1m"

echo -e "${RED}This is red${RESET}"
echo -e "${GREEN}This is green${RESET}"
echo -e "${BLUE}${BOLD}This is bold blue${RESET}"

function print(){
	local readonly text="$1"
	local readonly option1="$2"
	if [$# -eq 2]; then
		echo -e "${option1}${text}${RESET}"
	else
		local readonly option2="$3"

		echo -e "${option1}${option2}${text}${RESET}"
	fi
}

print "LOL" ${RED} ${BOLD}

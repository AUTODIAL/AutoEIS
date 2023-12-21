#!/bin/bash -i

DONE_MSG='\033[0;32mSUCCESS!\033[0m'
FAILED_MSG='\033[31mFAIL!\033[0m'

function check_julia_installed {
    source ~/.bashrc
    if julia -e 'println("Hello world")' &> /dev/null; then
        return 0
    else
        return 1
    fi
}

function is_juliaup_functional {
    source ~/.bashrc
    juliaup status &> /dev/null
    return $?
}

function check_juliaup_exists {
    if is_juliaup_functional; then
        return 0
    else
        return 1
    fi
}

function install_julia_via_juliaup {
    if juliaup add release &> /dev/null; then
        return 0
    else
        return 1
    fi
}

function uninstall_juliaup {
    expect -c 'spawn juliaup self uninstall; expect "Do you really want to uninstall Julia?" {send "y\r"}; expect eof' &> /dev/null
    if [ $? -eq 0 ]; then
	source ~/.bashrc
        return 0
    else
        return 1
    fi
}

function install_juliaup {
    curl -fsSL https://install.julialang.org | sh -s -- --default-channel release -y &> /dev/null
    is_juliaup_functional
    if [ $? -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

function report_status {
    if [ $1 -eq 0 ]; then
        echo -e " ... $DONE_MSG"
    else
        echo -e " ... $FAILED_MSG"
    fi
}

echo -n "Checking if Julia is already installed"
if check_julia_installed; then
    report_status 0
    exit 0
else
    report_status 1
fi

echo -n "Checking if juliaup is available and functional"
if check_juliaup_exists; then
    report_status 0
    echo -n "Attempting to install Julia using juliaup"
    if install_julia_via_juliaup; then
        report_status 0
        echo -n "Checking if Julia is now installed"
        if check_julia_installed; then
            report_status 0
            exit 0
        else
            report_status 1
        fi
    else
        report_status 1
        echo -n "Uninstalling problematic juliaup"
        if uninstall_juliaup; then
            report_status 0
        else
            report_status 1
            echo "Manual intervention required. Please uninstall juliaup manually."
            exit 1
        fi
    fi
else
    report_status 1
fi

echo -n "Attempting to install juliaup"
if install_juliaup; then
    report_status 0
    echo -n "Installing Julia using the newly installed juliaup"
    if install_julia_via_juliaup; then
        report_status 0
    else
        report_status 1
        echo "Failed to install Julia using juliaup."
        exit 1
    fi
else
    report_status 1
    echo "Failed to install juliaup."
    exit 1
fi

echo -n "Final check to confirm if Julia is installed"
if check_julia_installed; then
    report_status 0
else
    report_status 1
    echo "Installation process completed, but Julia doesn't seem to be functional. Manual intervention may be required."
    exit 1
fi

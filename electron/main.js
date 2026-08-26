const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

let dashProcess;
let mainWindow;

function dashArguments() {
  // Electron itself consumes the first arguments in development mode.
  return app.isPackaged ? process.argv.slice(1) : process.argv.slice(2);
}

function portFrom(argumentsList) {
  const portIndex = argumentsList.findIndex((argument) => argument === "--port" || argument === "-p");
  if (portIndex >= 0 && argumentsList[portIndex + 1]) {
    const port = Number(argumentsList[portIndex + 1]);
    if (Number.isInteger(port) && port > 0 && port < 65536) {
      return port;
    }
  }
  return 8888;
}

function executablePath() {
  const directory = app.isPackaged
    ? path.join(process.resourcesPath, "prof-dash-gui")
    : path.join(__dirname, "..", "dist", "prof-dash-gui");
  const executableName = process.platform === "win32" ? "prof-dash-gui.exe" : "prof-dash-gui";
  return path.join(directory, executableName);
}

function waitForServer(port, child) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    child.once("error", reject);
    const check = () => {
      if (child.exitCode !== null) {
        reject(new Error(`The Dash executable exited with code ${child.exitCode}.`));
        return;
      }
      const request = http.get(`http://127.0.0.1:${port}/`, (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      request.on("error", retry);
      request.setTimeout(1000, () => request.destroy());
    };
    const retry = () => {
      attempts += 1;
      if (attempts >= 120) {
        reject(new Error(`The Dash server did not start on port ${port}.`));
        return;
      }
      setTimeout(check, 250);
    };
    check();
  });
}

async function createWindow() {
  const argumentsList = dashArguments();
  const port = portFrom(argumentsList);
  const executable = executablePath();

  dashProcess = spawn(executable, argumentsList, {
    // Preserve the caller's working directory so relative config/checkpoint
    // paths behave like they do when invoking prof-dash-gui directly.
    cwd: process.cwd(),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  dashProcess.on("error", (error) => console.error(`[prof-dash-gui] ${error.message}`));
  dashProcess.stdout.on("data", (data) => console.log(`[prof-dash-gui] ${data}`.trimEnd()));
  dashProcess.stderr.on("data", (data) => console.error(`[prof-dash-gui] ${data}`.trimEnd()));

  try {
    await waitForServer(port, dashProcess);
  } catch (error) {
    dialog.showErrorBox("Professor Dash GUI", `${error.message}\n\nExecutable: ${executable}`);
    app.quit();
    return;
  }

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 1000,
    minWidth: 900,
    minHeight: 700,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  await mainWindow.loadURL(`http://127.0.0.1:${port}/`);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => app.quit());

app.on("before-quit", () => {
  if (dashProcess && !dashProcess.killed) {
    dashProcess.kill();
  }
});

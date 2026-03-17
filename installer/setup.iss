; EPM Installer - Inno Setup Script
; Wraps installer.ps1 into a standalone .exe
; Requirements: Inno Setup 6+ (https://jrsoftware.org/isinfo.php)

[Setup]
AppName=EPM - Electricity Planning Model
AppVersion=1.0
AppPublisher=ESMAP - World Bank Group
AppPublisherURL=https://github.com/ESMAP-World-Bank-Group/EPM
DefaultDirName={autopf}\EPM
DisableDirPage=yes
OutputDir=dist
OutputBaseFilename=EPM_Setup
SetupIconFile=
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
WizardStyle=modern
WizardSmallImageFile=
DisableProgramGroupPage=yes
UninstallDisplayName=EPM - Electricity Planning Model

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "installer.ps1"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NonInteractive -File ""{tmp}\installer.ps1"""; \
  Description: "Run EPM setup"; \
  Flags: runhidden waituntilterminated

[Messages]
WelcomeLabel1=Welcome to EPM Setup
WelcomeLabel2=This wizard will install the Electricity Planning Model (EPM) on your computer.%n%nIt will:%n  - Clone the EPM repository from GitHub%n  - Install the Python environment%n  - Create a desktop launcher for the dashboard%n%nGAMS must be installed separately with a valid license.
FinishedHeadingLabel=EPM Setup Complete
FinishedLabel=EPM has been installed successfully.%n%nDouble-click "Launch EPM Dashboard" on your Desktop to start.

; Windows installer for Claim-Aware RAG (Inno Setup 6).
; Build: python scripts/build_release.py   (or: ISCC installer\ClaimAwareRAG.iss)
;
; Installs per user (no administrator rights) into %LOCALAPPDATA%\Programs\ClaimAwareRAG,
; adds Start menu and desktop icons and an entry in Settings > Apps, then runs install.bat
; to set up Python, PyTorch (GPU build when an NVIDIA GPU is present) and the models.
; Optionally installs Ollama and qwen3:8b for AI-written answers.

#define AppName "Claim-Aware RAG"
#define AppVersion "0.3.0"
#define AppExe "{app}\.venv\Scripts\carag-desktop.exe"
#define AppIcon "{app}\carag\server\static\icon.ico"
#define AiModel "qwen3:8b"

[Setup]
AppId={{8F3B2C1E-6A4D-4E8B-9C2F-1D7A5B3E9F01}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
AppPublisherURL=https://github.com/manishreddy1767/claim-aware-adaptive-rag
DefaultDirName={localappdata}\Programs\ClaimAwareRAG
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ClaimAwareRAG-Setup
SetupIconFile=..\carag\server\static\icon.ico
UninstallDisplayIcon={#AppIcon}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
AppMutex=ClaimAwareRAGDesktop
CloseApplications=no

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nIt answers questions about your own documents and checks every answer against them. Everything runs on this computer: your graphics card (or processor) does the work and your documents stay on your disk.%n%nSetup downloads about 5 GB of components (Python, PyTorch and language models), so it needs an internet connection and can take 10-30 minutes.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Icons:"
Name: "ai"; Description: "Install a local AI model for AI-written answers (Ollama and {#AiModel}, about 5 GB more)"; GroupDescription: "Optional:"

[Files]
Source: "..\carag\*"; DestDir: "{app}\carag"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\install.bat"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{#AppExe}"; WorkingDir: "{app}"; IconFilename: "{#AppIcon}"; AppUserModelID: "ClaimAwareRAG.Desktop"; Comment: "Ask questions about your documents"
Name: "{autodesktop}\{#AppName}"; Filename: "{#AppExe}"; WorkingDir: "{app}"; IconFilename: "{#AppIcon}"; AppUserModelID: "ClaimAwareRAG.Desktop"; Tasks: desktopicon

[Run]
Filename: "{#AppExe}"; Description: "Start {#AppName}"; Flags: postinstall nowait skipifsilent; Check: ComponentsOK

[UninstallDelete]
; Created after installation (Python environment, build metadata, caches); not your documents.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\build"
Type: filesandordirs; Name: "{app}\claim_aware_rag.egg-info"
Type: filesandordirs; Name: "{app}\carag"
Type: dirifempty; Name: "{app}"

[Code]
var
  ComponentsInstalled: Boolean;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Args: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    Args := '/c ""' + ExpandConstant('{app}\install.bat') + '" --no-shortcut --no-pause';
    if WizardIsTaskSelected('ai') then
      Args := Args + ' --ai {#AiModel}';
    Args := Args + '"';
    WizardForm.StatusLabel.Caption := 'Installing Python, PyTorch and the language models. A window shows the progress; this can take 10-30 minutes...';
    { The console window stays visible so the long downloads show their progress. }
    ComponentsInstalled := Exec(ExpandConstant('{cmd}'), Args, ExpandConstant('{app}'), SW_SHOWNORMAL,
                                ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
    if not ComponentsInstalled then
      SuppressibleMsgBox('Some components could not be installed (see the window that showed the progress).' + #13#10 + #13#10 +
        'Check your internet connection, then run install.bat in' + #13#10 + ExpandConstant('{app}') + #13#10 +
        'or run this setup again.', mbError, MB_OK, IDOK);
  end;
end;

function ComponentsOK: Boolean;
begin
  { Only offer to start the application when its components installed. }
  Result := ComponentsInstalled;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
    { The Python environment has paths longer than 260 characters, which the uninstaller's own
      deletion cannot remove; rmdir with the \\?\ prefix can. }
    Exec(ExpandConstant('{cmd}'), '/c rmdir /s /q "\\?\' + ExpandConstant('{app}\.venv') + '"', '',
         SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\ClaimAwareRAG');
    if DirExists(DataDir) then
      if SuppressibleMsgBox('Also delete your accounts, documents and question history?' + #13#10 + #13#10 +
                            DataDir + #13#10 + #13#10 + 'Choose No to keep them for a later installation.',
                            mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;

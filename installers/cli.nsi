; Tanrenai CLI Windows Installer
; Built with NSIS (Nullsoft Scriptable Install System)

!include "MUI2.nsh"
!include "EnvVarUpdate.nsh"

; ── Metadata ─────────────────────────────────────────────────────────────

Name "Tanrenai CLI"
OutFile "tanrenai-cli-windows-amd64-setup.exe"
InstallDir "$LOCALAPPDATA\Tanrenai\CLI"
InstallDirRegKey HKCU "Software\Tanrenai\CLI" "InstallDir"
RequestExecutionLevel user

!define MUI_ICON "logo.ico"
!define MUI_UNICON "logo.ico"

; ── UI ───────────────────────────────────────────────────────────────────

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Tanrenai CLI Setup"
!define MUI_WELCOMEPAGE_TEXT "This will install the Tanrenai CLI on your computer.$\r$\n$\r$\nThe installer will add tanrenai to your PATH so you can run it from any terminal."

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ── Install ──────────────────────────────────────────────────────────────

Section "Tanrenai CLI" SecMain
  SetOutPath "$INSTDIR"

  ; Main executable
  File "tanrenai-cli.exe"

  ; Bundled llama-server and DLLs
  SetOutPath "$INSTDIR\bin"
  File /nonfatal "bin\*.*"

  ; Back to install root
  SetOutPath "$INSTDIR"

  ; Add to user PATH
  ${EnvVarUpdate} $0 "PATH" "A" "HKCU" "$INSTDIR"

  ; Notify running applications that PATH changed
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Registry entries for Add/Remove Programs
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "DisplayName" "Tanrenai CLI"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "Publisher" "ThatCatDev"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "DisplayVersion" "${VERSION}"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI" \
    "NoRepair" 1

  ; Save install dir
  WriteRegStr HKCU "Software\Tanrenai\CLI" "InstallDir" "$INSTDIR"
SectionEnd

; ── Uninstall ────────────────────────────────────────────────────────────

Section "Uninstall"
  ; Remove from PATH
  ${un.EnvVarUpdate} $0 "PATH" "R" "HKCU" "$INSTDIR"
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Remove files
  RMDir /r "$INSTDIR\bin"
  Delete "$INSTDIR\tanrenai-cli.exe"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"

  ; Remove registry
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiCLI"
  DeleteRegKey HKCU "Software\Tanrenai\CLI"
SectionEnd

; Tanrenai Desktop Windows Installer
; Built with NSIS (Nullsoft Scriptable Install System)

!include "MUI2.nsh"
!include "EnvVarUpdate.nsh"

; ── Metadata ─────────────────────────────────────────────────────────────

Name "Tanrenai Desktop"
OutFile "tanrenai-desktop-windows-amd64-setup.exe"
InstallDir "$LOCALAPPDATA\Tanrenai\Desktop"
InstallDirRegKey HKCU "Software\Tanrenai\Desktop" "InstallDir"
RequestExecutionLevel user

!define MUI_ICON "logo.ico"
!define MUI_UNICON "logo.ico"

; ── UI ───────────────────────────────────────────────────────────────────

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Tanrenai Desktop Setup"
!define MUI_WELCOMEPAGE_TEXT "This will install Tanrenai Desktop on your computer.$\r$\n$\r$\nA Start Menu shortcut and PATH entry will be created."

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ── Install ──────────────────────────────────────────────────────────────

Section "Tanrenai Desktop" SecMain
  SetOutPath "$INSTDIR"

  ; Main executable + GTK DLLs (all in root)
  File "tanrenai-desktop.exe"
  File /nonfatal "*.dll"

  ; Bundled llama-server and CUDA DLLs
  SetOutPath "$INSTDIR\bin"
  File /nonfatal "bin\*.*"

  ; GTK resources
  SetOutPath "$INSTDIR\share\glib-2.0\schemas"
  File /nonfatal /r "share\glib-2.0\schemas\*.*"

  SetOutPath "$INSTDIR\share\icons"
  File /nonfatal /r "share\icons\*.*"

  SetOutPath "$INSTDIR\share\gtk-4.0"
  File /nonfatal /r "share\gtk-4.0\*.*"

  SetOutPath "$INSTDIR\lib"
  File /nonfatal /r "lib\*.*"

  ; Back to install root
  SetOutPath "$INSTDIR"

  ; Add to user PATH
  ${EnvVarUpdate} $0 "PATH" "A" "HKCU" "$INSTDIR"

  ; Notify running applications that PATH changed
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Start Menu shortcut
  CreateDirectory "$SMPROGRAMS\Tanrenai"
  CreateShortCut "$SMPROGRAMS\Tanrenai\Tanrenai Desktop.lnk" "$INSTDIR\tanrenai-desktop.exe"
  CreateShortCut "$SMPROGRAMS\Tanrenai\Uninstall Tanrenai Desktop.lnk" "$INSTDIR\uninstall.exe"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Registry entries for Add/Remove Programs
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "DisplayName" "Tanrenai Desktop"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "Publisher" "ThatCatDev"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "DisplayVersion" "${VERSION}"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop" \
    "NoRepair" 1

  ; Save install dir
  WriteRegStr HKCU "Software\Tanrenai\Desktop" "InstallDir" "$INSTDIR"
SectionEnd

; ── Uninstall ────────────────────────────────────────────────────────────

Section "Uninstall"
  ; Remove from PATH
  ${un.EnvVarUpdate} $0 "PATH" "R" "HKCU" "$INSTDIR"
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Remove Start Menu
  Delete "$SMPROGRAMS\Tanrenai\Tanrenai Desktop.lnk"
  Delete "$SMPROGRAMS\Tanrenai\Uninstall Tanrenai Desktop.lnk"
  RMDir "$SMPROGRAMS\Tanrenai"

  ; Remove files
  RMDir /r "$INSTDIR\bin"
  RMDir /r "$INSTDIR\share"
  RMDir /r "$INSTDIR\lib"
  Delete "$INSTDIR\tanrenai-desktop.exe"
  Delete "$INSTDIR\*.dll"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"

  ; Remove registry
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TanrenaiDesktop"
  DeleteRegKey HKCU "Software\Tanrenai\Desktop"
SectionEnd

/**
 *  EnvVarUpdate.nsh
 *    : Environmental Variables: append, prepend, and remove entries
 *
 *     WARNING: If you use StrFunc.nsh header then include it before this file
 *              with all required definitions. This is to avoid conflicts
 *
 *  Usage:
 *    ${EnvVarUpdate} "ResultVar" "EnvVarName" "Action" "RegLoc" "PathString"
 *
 *  Credits:
 *    Version 1.0
 *    Written by murber
 *    Modified by Wikipedia:User:Dumpydooby
 *    Modified by Wikipedia:User:Wikipedia:User:WikiNick
 *    Adapted from AddToPath and RemoveFromPath
 *    http://nsis.sourceforge.net/Environmental_Variables:_append%2C_prepend%2C_and_remove_entries
 *
 *  Parameters:
 *    ResultVar   - Result: new value of env var (not currently used)
 *    EnvVarName  - Name of the environment variable (e.g., "PATH")
 *    Action      - "A" = Append, "P" = Prepend, "R" = Remove
 *    RegLoc      - "HKLM" = all users, "HKCU" = current user
 *    PathString  - The string to add or remove
 */

!ifndef ENVVARUPDATE_FUNCTION
!define ENVVARUPDATE_FUNCTION
!verbose push
!verbose 3
!include "LogicLib.nsh"
!include "WinMessages.nsh"

!macro _EnvVarUpdateConstructor ResultVar EnvVarName Action RegLoc PathString
  Push "${PathString}"
  Push "${Action}"
  Push "${EnvVarName}"
  Push "${RegLoc}"
  Call EnvVarUpdate
  Pop "${ResultVar}"
!macroend
!define EnvVarUpdate '!insertmacro "_EnvVarUpdateConstructor"'

!macro _un.EnvVarUpdateConstructor ResultVar EnvVarName Action RegLoc PathString
  Push "${PathString}"
  Push "${Action}"
  Push "${EnvVarName}"
  Push "${RegLoc}"
  Call un.EnvVarUpdate
  Pop "${ResultVar}"
!macroend
!define un.EnvVarUpdate '!insertmacro "_un.EnvVarUpdateConstructor"'

!macro EnvVarUpdate_func un
Function ${un}EnvVarUpdate

  Exch $R0 ; RegLoc
  Exch
  Exch $R1 ; EnvVarName
  Exch 2
  Exch $R2 ; Action
  Exch 3
  Exch $R3 ; PathString

  Push $R4 ; Current path value
  Push $R5 ; Temp
  Push $R6 ; Length of PathString
  Push $R7 ; Temp path for iteration
  Push $R8 ; Result

  ; Read current value
  ${If} $R0 == "HKLM"
    ReadRegStr $R4 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $R1
  ${ElseIf} $R0 == "HKCU"
    ReadRegStr $R4 HKCU "Environment" $R1
  ${EndIf}

  ; Get length of PathString
  StrLen $R6 $R3

  ; === REMOVE (always done first, even for Append/Prepend to avoid dupes) ===
  StrCpy $R8 ""
  StrCpy $R7 $R4

  ${If} $R7 != ""
  loop_remove:
    ; Find next semicolon
    StrCpy $R5 ""
    ${Do}
      StrCpy $R5 $R7 1
      ${If} $R5 == ""
      ${OrIf} $R5 == ";"
        ${ExitDo}
      ${EndIf}
      StrLen $R5 $R7
      ${If} $R5 <= 0
        ${ExitDo}
      ${EndIf}
      ; Extract char and advance
      StrCpy $R5 $R7 1
      StrCpy $R7 $R7 "" 1
    ${Loop}

    ; Get the segment before the semicolon
    StrLen $R5 $R7
    StrLen $R6 $R4
    IntOp $R6 $R6 - $R5
    ${If} $R5 != ""
      IntOp $R6 $R6 - 1 ; subtract the semicolon
    ${EndIf}
    StrCpy $R5 $R4 $R6

    ; Compare segment to PathString (case insensitive)
    ${If} $R5 != $R3
      ${If} $R8 != ""
        StrCpy $R8 "$R8;$R5"
      ${Else}
        StrCpy $R8 $R5
      ${EndIf}
    ${EndIf}

    ; Advance past semicolon
    StrCpy $R4 $R7
    ${If} $R4 != ""
      StrCpy $R5 $R4 1
      ${If} $R5 == ";"
        StrCpy $R4 $R4 "" 1
      ${EndIf}
      StrCpy $R7 $R4
      Goto loop_remove
    ${EndIf}
  ${EndIf}

  ; === APPEND or PREPEND ===
  ${If} $R2 == "A"
    ${If} $R8 != ""
      StrCpy $R8 "$R8;$R3"
    ${Else}
      StrCpy $R8 $R3
    ${EndIf}
  ${ElseIf} $R2 == "P"
    ${If} $R8 != ""
      StrCpy $R8 "$R3;$R8"
    ${Else}
      StrCpy $R8 $R3
    ${EndIf}
  ${EndIf}
  ; If Action == "R", R8 already has the entry removed

  ; Write back
  ${If} $R0 == "HKLM"
    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" $R1 $R8
  ${ElseIf} $R0 == "HKCU"
    WriteRegExpandStr HKCU "Environment" $R1 $R8
  ${EndIf}

  Pop $R8
  Pop $R7
  Pop $R6
  Pop $R5
  Pop $R4
  Pop $R3
  Pop $R2
  Pop $R1
  Pop $R0
  Push $R8
FunctionEnd
!macroend

!insertmacro EnvVarUpdate_func ""
!insertmacro EnvVarUpdate_func "un."

!verbose pop
!endif

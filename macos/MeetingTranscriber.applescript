-- Meeting Transcriber.app: a Mac front end for the `transcribe` command.
-- Open the app to pick a recording, or drop audio or video files on its icon.
-- The run happens in a Terminal window, so progress and errors stay visible.
-- Build it with macos/build.sh.

property appTitle : "Meeting Transcriber"

on homePath()
	return POSIX path of (path to home folder)
end homePath

on configDir()
	return homePath() & ".config/meeting-transcriber/"
end configDir

on isConfigured()
	try
		do shell script "test -f " & quoted form of (configDir() & "config.yaml")
		return true
	on error
		return false
	end try
end isConfigured

-- Write a one-off .command file and open it in Terminal, so the user sees progress.
-- Opening a .command file needs no Automation permission.
on runInTerminal(body)
	set jobDir to homePath() & "Library/Caches/meeting-transcriber-app/"
	do shell script "mkdir -p " & quoted form of jobDir
	set jobFile to do shell script "mktemp " & quoted form of (jobDir & "job-XXXXXX") & " && true"
	set scriptText to "#!/bin/zsh" & linefeed & ¬
		"rm -f \"$0\"" & linefeed & ¬
		"export PATH=\"$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH\"" & linefeed & ¬
		"clear" & linefeed & ¬
		"printf '\\e]0;Meeting Transcriber\\a'" & linefeed & ¬
		body & linefeed & ¬
		"echo" & linefeed & "read -rs '?Press Enter to close this window.'" & linefeed & "exit" & linefeed
	set cmdFile to jobFile & ".command"
	set fh to open for access (POSIX file cmdFile) with write permission
	set eof of fh to 0
	write scriptText to fh as «class utf8»
	close access fh
	do shell script "rm -f " & quoted form of jobFile & "; chmod +x " & quoted form of cmdFile & "; open -a Terminal " & quoted form of cmdFile
end runInTerminal

on runSetup()
	set body to "echo 'First-time setup for Meeting Transcriber.'" & linefeed & ¬
		"echo 'You need an ElevenLabs API key (speech to text) and a key for your LLM provider.'" & linefeed & ¬
		"echo 'Example: provider openrouter, model google/gemini-2.5-flash. Keys are typed hidden and stay in ~/.config/meeting-transcriber/.env'" & linefeed & ¬
		"echo" & linefeed & ¬
		"transcribe init && echo && transcribe doctor"
	runInTerminal(body)
end runSetup

on transcribeBody(fileArgs, speakerArg)
	return "out=$(mktemp)" & linefeed & ¬
		"if transcribe" & fileArgs & speakerArg & " --json > \"$out\"; then" & linefeed & ¬
		"  t=$(plutil -extract transcript raw -o - \"$out\")" & linefeed & ¬
		"  s=$(plutil -extract short raw -o - \"$out\" 2>/dev/null)" & linefeed & ¬
		"  echo; echo '──────── Short summary ────────'; [ -f \"$s\" ] && cat \"$s\"; echo" & linefeed & ¬
		"  echo; echo \"Saved in: $(dirname \"$t\")\"" & linefeed & ¬
		"  open \"$(dirname \"$t\")\"" & linefeed & ¬
		"  osascript -e 'display notification \"Transcript and summary are ready.\" with title \"Meeting Transcriber\" sound name \"Glass\"'" & linefeed & ¬
		"else" & linefeed & ¬
		"  echo; echo 'Transcription failed. See the message above.'" & linefeed & ¬
		"  osascript -e 'display notification \"Transcription failed. See the Terminal window.\" with title \"Meeting Transcriber\"'" & linefeed & ¬
		"fi" & linefeed & "rm -f \"$out\""
end transcribeBody

on transcribeFiles(theFiles)
	if not isConfigured() then
		display dialog "Meeting Transcriber is not set up yet. It needs your ElevenLabs and LLM API keys first." buttons {"Cancel", "Set up now"} default button 2 with title appTitle with icon note
		runSetup()
		return
	end if
	set fileArgs to ""
	set names to ""
	repeat with f in theFiles
		set p to POSIX path of f
		set fileArgs to fileArgs & " " & quoted form of p
		set names to names & "  " & (do shell script "basename " & quoted form of p) & return
	end repeat
	set reply to display dialog "Transcribing:" & return & names & return & "Who was on the call? For example: Alice, Sam: client" & return & "Leave empty to let it work out the names." default answer "" buttons {"Cancel", "Transcribe"} default button 2 with title appTitle with icon note
	set speakers to text returned of reply
	set speakerArg to ""
	if speakers is not "" then set speakerArg to " -s " & quoted form of speakers
	set body to transcribeBody(fileArgs, speakerArg)
	runInTerminal(body)
end transcribeFiles

on chooseAndTranscribe()
	set theFiles to choose file with prompt "Choose a recording (audio or video). You can pick several parts of one call." of type {"public.audio", "public.movie", "public.audiovisual-content"} default location (path to downloads folder) with multiple selections allowed
	transcribeFiles(theFiles)
end chooseAndTranscribe

on moreMenu()
	set opts to {"Check setup", "Edit API keys", "Edit config (models, projects)", "Open meetings folder", "Record a call (needs BlackHole)", "Run first-time setup", "Update Meeting Transcriber"}
	set pick to choose from list opts with title appTitle with prompt "What would you like to do?" default items {"Check setup"}
	if pick is false then return
	set pick to item 1 of pick
	if pick is "Check setup" then
		runInTerminal("transcribe doctor")
	else if pick is "Edit API keys" then
		if not isConfigured() then return runSetup()
		do shell script "open -e " & quoted form of (configDir() & ".env")
	else if pick is "Edit config (models, projects)" then
		if not isConfigured() then return runSetup()
		do shell script "open -e " & quoted form of (configDir() & "config.yaml")
	else if pick is "Open meetings folder" then
		do shell script "mkdir -p ~/Meetings && open ~/Meetings"
	else if pick is "Record a call (needs BlackHole)" then
		runInTerminal("echo 'Recording. Press Ctrl-C to stop; the file goes to your inbox.'; echo 'Then open Meeting Transcriber and choose the recording.'; echo; transcribe record")
	else if pick is "Run first-time setup" then
		runSetup()
	else if pick is "Update Meeting Transcriber" then
		runInTerminal("uv tool upgrade meeting-transcriber")
	end if
end moreMenu

on run
	if not isConfigured() then
		set r to button returned of (display dialog "Welcome to Meeting Transcriber." & return & return & "It turns a call recording into a transcript with named speakers plus a summary." & return & return & "First, set it up with your ElevenLabs and LLM API keys." buttons {"Quit", "Set up"} default button 2 with title appTitle with icon note)
		if r is "Set up" then runSetup()
		return
	end if
	set r to button returned of (display dialog "Turn a call recording into a transcript with named speakers and a summary." & return & return & "Tip: drop audio or video files on the app icon." buttons {"More…", "Quit", "Transcribe a recording…"} default button 3 cancel button 2 with title appTitle with icon note)
	if r is "Transcribe a recording…" then
		chooseAndTranscribe()
	else if r is "More…" then
		moreMenu()
	end if
end run

on open theFiles
	transcribeFiles(theFiles)
end open

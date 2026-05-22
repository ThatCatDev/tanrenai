module github.com/ThatCatDev/tanrenai/desktop

go 1.25.0

require (
	github.com/ThatCatDev/tanrenai-gpu v1.4.0
	github.com/ThatCatDev/tanrenai/server v0.0.0
	github.com/ThatCatDev/tanrenai/shared v0.0.0
	github.com/diamondburned/gotk4-adwaita/pkg v0.0.0-20250703085740-f81761ef0e0d
	github.com/diamondburned/gotk4/pkg v0.3.2-0.20250703063411-16654385f59a
)

require (
	github.com/KarpelesLab/weak v0.1.1 // indirect
	github.com/PuerkitoBio/goquery v1.11.0 // indirect
	github.com/andybalholm/cascadia v1.3.3 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/philippgille/chromem-go v0.7.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	go4.org/unsafe/assume-no-moving-gc v0.0.0-20231121144256-b99613f794b6 // indirect
	golang.org/x/net v0.50.0 // indirect
	golang.org/x/sync v0.10.0 // indirect
)

replace (
	github.com/ThatCatDev/tanrenai/server => ../../server
	github.com/ThatCatDev/tanrenai/shared => ../shared
)

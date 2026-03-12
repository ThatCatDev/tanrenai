package cmd

import (
	"math"
	"strings"
	"time"

	"github.com/rivo/tview"
)

// ASCII art — 鍛錬 (tanren) = forging/tempering.
// Hammer rotates around a pivot (grip point) to strike the anvil.
// Pre-rendered at 7 angles from 0° (strike) to 30° (raised).
// { = head, # = handle. All frames are 11 lines tall.

var hammerFrames = [][]string{
	// Frame 0: 0° (strike)
	{
		`                     #####`,
		`                   #######{`,
		`    {{          #######{###`,
		`  {{{{{{{{   #######{#####`,
		` {{{{{{{{{{{{####{######`,
		`{{{{{{{{{{{{{#{######`,
		` {{{{{{{{{{{{{{###`,
		`  {{{{{{{{{{{{{`,
		`   {{{{{{{{{{{`,
		`    {{{{{{{{{`,
		`     {{{{{`,
	},
	// Frame 1: 5°
	{
		`                      ####`,
		`    {{{             #######`,
		`  {{{{{{{{{     ##########{{`,
		` {{{{{{{{{{{{###########{##`,
		`{{{{{{{{{{{{{{###{{######`,
		` {{{{{{{{{{{{{{{#####`,
		`  {{{{{{{{{{{{{#`,
		`   {{{{{{{{{{{`,
		`    {{{{{{{{{`,
		`      {{{{`,
		``,
	},
	// Frame 2: 10°
	{
		`   {{{{{`,
		`  {{{{{{{{{         #######`,
		` {{{{{{{{{{{  #############{`,
		` {{{{{{{{{{{{{#######{##{##{`,
		`  {{{{{{{{{{{##{#{{##{#####`,
		`   {{{{{{{{{{{{#######`,
		`   {{{{{{{{{{{{`,
		`    {{{{{{{{{`,
		`        {{`,
		``,
		``,
	},
	// Frame 3: 15°
	{
		`   {{{{{`,
		`  {{{{{{{`,
		` {{{{{{{{{{           ##`,
		` {{{{{{{{{{{{ #############`,
		`  {{{{{{{{{{{{#############{`,
		`   {{{{{{{{{{####{{#{{##{##{`,
		`   {{{{{{{{{{{{########## #`,
		`    {{{{{{{{{ #`,
		`      {{{{{{{`,
		``,
		``,
	},
	// Frame 4: 20°
	{
		`  {{{{{{`,
		`{{{{{{{{{`,
		` {{{{{{{{{{`,
		` {{{{{{{{{{           #`,
		`  {{{{{{{{{{{#############`,
		`  {{{{{{{{{{###############`,
		`   {{{{{{{{{{{{##{##{##{##{`,
		`   {{{{{{{{{{#############`,
		`      {{{{{{`,
		``,
		``,
	},
	// Frame 5: 25°
	{
		`  {{{{{`,
		`{{{{{{{`,
		` {{{{{{{{{`,
		` {{{{{{{{{{`,
		`  {{{{{{{{{{ #`,
		`  {{{{{{{{{{{##### ####`,
		`  {{{{{{{{{{#############`,
		`   {{{{{{{{{{{##{#########`,
		`     {{{{{{{{# #####{#{{##{`,
		`       {  {        # #####`,
		``,
	},
	// Frame 6: 30° (fully raised)
	{
		` {{{{{{`,
		` {{{{{{{`,
		` {{{{{{{{`,
		` {{{{{{{{{{`,
		`  {{{{{{{{{`,
		`  {{{{{{{{{{{#`,
		`  {{{{{{{{{{###### # #`,
		`   {{{{{{{{{#{##########`,
		`     {{{{{{{####{##{######`,
		`         {{    ####{##{###`,
		`                     ####{`,
	},
}

var anvilLines = []string{
	`%#####{#{{{{{#{{{{######{#{{{{{{{{{{{{{{`,
	`  #{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{##`,
	`     #{{{{{{{{{{{{{{{{{{{{{{{##`,
	`            ##{{{{{{{{{{{{{`,
	`              {{{{{{}]]]]]]`,
	`           %#{{#{{{{###{}](((]`,
	`         #{{{{{{{{{{{{{{{{}[}}}{{`,
	`   #{{{{}}}}}}}}}}}}}}}}}}}}}}}}{{{{{{#`,
}

// Spark lines inserted at the hammer-anvil junction on strike.
var sparkSets = [][]string{
	{`          ✦    *         ·    ✦`, `            ·       ✦    *`},
	{`              *  ·    ✦       `, `        ✦         *    ·   ✦`},
}

// Braille spinner characters for the loading indicator.
var spinnerChars = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

const spinnerTickInterval = 60 * time.Millisecond

// Colors
const (
	headColor   = "[#cc5500::-]" // forged iron orange (head)
	handleColor = "[#cc2200::-]" // red handle
	anvilColor  = "[#667788::-]" // dark steel
	sparkColor  = "[#ffcc00::b]" // bright yellow bold
	fadeColor   = "[#664400::-]" // dim amber
	colorReset  = "[-:-:-]"
)

var numHammerFrames = len(hammerFrames)

// animFrames is the precomputed sequence for the full animation cycle.
var animFrames []animFrame

type animFrame struct {
	hammerIdx int
	spark     bool
	fading    bool
}

func init() {
	animFrames = buildAnimFrames()
}

func easeInCubic(t float64) float64 {
	return t * t * t
}

func easeOutCubic(t float64) float64 {
	t = 1 - t
	return 1 - t*t*t
}

// buildAnimFrames generates the full animation cycle:
//  1. Hold raised
//  2. Ease-in swing down (slow → fast into impact)
//  3. Impact hold with sparks
//  4. Ease-out swing back (fast → slow to raised)
func buildAnimFrames() []animFrame {
	var frames []animFrame
	maxIdx := numHammerFrames - 1

	// Phase 1: Hold raised
	for i := 0; i < 12; i++ {
		frames = append(frames, animFrame{hammerIdx: maxIdx})
	}

	// Phase 2: Ease-in swing down
	downSteps := 12
	for i := 0; i < downSteps; i++ {
		t := float64(i) / float64(downSteps-1)
		eased := easeInCubic(t)
		idx := maxIdx - int(math.Round(eased*float64(maxIdx)))
		frames = append(frames, animFrame{hammerIdx: idx})
	}

	// Phase 3: Impact with sparks
	for i := 0; i < 6; i++ {
		frames = append(frames, animFrame{hammerIdx: 0, spark: true})
	}
	for i := 0; i < 3; i++ {
		frames = append(frames, animFrame{hammerIdx: 0, fading: true})
	}

	// Phase 4: Ease-out swing back
	upSteps := 12
	for i := 0; i < upSteps; i++ {
		t := float64(i) / float64(upSteps-1)
		eased := easeOutCubic(t)
		idx := int(math.Round(eased * float64(maxIdx)))
		frames = append(frames, animFrame{hammerIdx: idx})
	}

	return frames
}

// colorizeHammer applies color tags to a hammer frame line.
func colorizeHammer(line string) string {
	escaped := tview.Escape(line)
	var b strings.Builder
	for _, r := range escaped {
		switch r {
		case '{':
			b.WriteString(headColor)
			b.WriteRune(r)
			b.WriteString(colorReset)
		case '#':
			b.WriteString(handleColor)
			b.WriteRune(r)
			b.WriteString(colorReset)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// colorizeAnvil applies color tags to an anvil line.
func colorizeAnvil(line string) string {
	escaped := tview.Escape(line)
	var b strings.Builder
	for _, r := range escaped {
		switch r {
		case '#', '%', '{', '}', ']', '(', '[':
			b.WriteString(anvilColor)
			b.WriteRune(r)
			b.WriteString(colorReset)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// startLoadingAnimation begins the anvil animation + spinner during startup.
func (t *tuiApp) startLoadingAnimation() {
	t.anvilFrame = 0
	t.anvilStop = make(chan struct{})
	stop := t.anvilStop

	go func() {
		ticker := time.NewTicker(spinnerTickInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				t.app.QueueUpdateDraw(func() {
					t.anvilFrame++
					t.refreshChatView()
				})
			}
		}
	}()
}

// stopLoadingAnimation stops the animation and clears it from the view.
func (t *tuiApp) stopLoadingAnimation() {
	if t.anvilStop != nil {
		close(t.anvilStop)
		t.anvilStop = nil
	}
	t.anvilFrame = -1
}

// renderLoadingArt returns the animated rotating hammer + static anvil.
func renderLoadingArt(tick int) string {
	if tick < 0 {
		return ""
	}

	frame := animFrames[tick%len(animFrames)]
	strikeCount := tick / len(animFrames)
	hammerFrame := hammerFrames[frame.hammerIdx]

	var lines []string
	lines = append(lines, "")

	pad := "  "

	for _, l := range hammerFrame {
		lines = append(lines, pad+colorizeHammer(l))
	}

	// Sparks between hammer and anvil
	if frame.spark {
		sparks := sparkSets[strikeCount%len(sparkSets)]
		for _, s := range sparks {
			lines = append(lines, pad+sparkColor+s+colorReset)
		}
	} else if frame.fading {
		sparks := sparkSets[(strikeCount+1)%len(sparkSets)]
		for _, s := range sparks {
			lines = append(lines, pad+fadeColor+s+colorReset)
		}
	}

	for _, l := range anvilLines {
		lines = append(lines, pad+colorizeAnvil(l))
	}

	// Spinner + text
	spinner := spinnerChars[tick%len(spinnerChars)]
	lines = append(lines, "")
	lines = append(lines, "  "+spinner+" [gray::-]Loading...[-::-]")

	return strings.Join(lines, "\n")
}

# TradingAgents UI Style Guide

This document defines UI patterns and components for the TradingAgents web frontend.

## Help Tooltips

Use the `HelpTooltip` component to provide contextual help for users. This displays a question mark icon that shows explanatory text on hover.

### When to Use

- Form fields that need explanation
- Technical terms (SMC concepts like FVG, OB, etc.)
- Settings/options that affect behavior
- Any UI element that might confuse users
- Numeric values that need context (percentages, ratios, strength indicators)

### Writing Effective Help Text

Help tooltips are a teaching tool. Users need continuous reinforcement to learn trading concepts. Write help text that **educates**, not just defines.

#### Principles

1. **Explain the "what", "why", AND "so what"**
   - Bad: "Order Block - institutional entry zone"
   - Good: "Order Block - the last candle before a strong move. If price returns here, it often bounces because institutions defend their entries."

2. **Be actionable - tell users what to expect and do**
   - Bad: "Strength: 65%"
   - Good: "Strength 65%: Moderate - may push price but less reliable. Consider waiting for confluence with other signals."

3. **Use concrete numbers and thresholds**
   - Bad: "High strength means it's more likely to work"
   - Good: "80-100% = Very strong, high probability. 50-79% = Moderate, less reliable. Below 50% = Weak, use caution."

4. **Explain implications for the user's position**
   - Bad: "Bullish FVG detected"
   - Good: "Bullish FVG - price imbalance zone. If price returns here, expect a bounce UP. Good area to place buy orders or move SL below."

5. **Make it directional when relevant**
   - For bullish zones: explain price likely to go UP
   - For bearish zones: explain price likely to go DOWN
   - Don't be generic when you can be specific

#### Examples

```tsx
// ❌ Bad - just a definition
<HelpTooltip content="ATR measures volatility" />

// ✅ Good - explains what it means for the user
<HelpTooltip content="ATR (14) measures average price movement over 14 candles. Higher ATR = more volatile market, use wider stops. Lower ATR = calmer market, can use tighter stops." />

// ❌ Bad - no context for the number
<HelpTooltip content="Strength indicates how strong the zone is" />

// ✅ Good - actionable thresholds
<HelpTooltip content="80-100% = Very strong, high probability of reaction. 50-79% = Moderate, wait for confirmation. Below 50% = Weak, don't rely on this zone alone." />

// ❌ Bad - generic
<HelpTooltip content="This is a support level" />

// ✅ Good - directional and actionable
<HelpTooltip content="Bullish OB at 4950 - if price drops to this level, expect buyers to step in and push price UP. Good place for a buy entry or to trail your stop loss just below." />
```

#### Dynamic Help Text

When the meaning changes based on context (bullish vs bearish), make the tooltip dynamic:

```tsx
<HelpTooltip
  content={zone.type === "bullish"
    ? "Bullish zone - price likely to bounce UP from here"
    : "Bearish zone - price likely to reject DOWN from here"
  }
/>
```

### Usage

```tsx
import { HelpTooltip, LabelWithHelp } from "@/components/ui/help-tooltip"

// Standalone help icon
<HelpTooltip content="Explanation text here" />

// With custom positioning
<HelpTooltip content="Explanation" side="right" align="start" />

// Smaller icon for inline use
<HelpTooltip content="Help text" iconClassName="h-3 w-3" />

// Label with integrated help
<LabelWithHelp help="What this field does" htmlFor="input-id">
  Field Label
</LabelWithHelp>
```

### Pattern: Label with Inline Help

```tsx
<div className="flex items-center gap-1">
  <label className="text-sm text-muted-foreground">Field Name</label>
  <HelpTooltip content="Explanation of this field" />
</div>
```

### Pattern: Checkbox with Help

```tsx
<div className="flex items-center gap-2">
  <Checkbox id="option" checked={value} onCheckedChange={onChange} />
  <label htmlFor="option" className="text-sm text-muted-foreground cursor-pointer">
    Option Name
  </label>
  <HelpTooltip content="What this option does" />
</div>
```

## DO NOT Use

- **HTML `title` attributes** - These have poor styling and accessibility
- **Underline decoration for hover hints** - Use HelpTooltip instead

## Badge Variants

Use semantic badge variants for trading context:

```tsx
// Buy/Bullish - green
<Badge variant="buy">BUY</Badge>

// Sell/Bearish - red
<Badge variant="sell">SELL</Badge>

// Neutral/inactive
<Badge variant="secondary">NEUTRAL</Badge>

// Status indicators
<Badge variant="outline">Active</Badge>
<Badge variant="destructive">Error</Badge>
```

## Color Semantics

| Color | Usage |
|-------|-------|
| Green (`text-green-500`, `bg-green-500`) | Bullish, buy, profit, support, positive |
| Red (`text-red-500`, `bg-red-500`) | Bearish, sell, loss, resistance, negative |
| Yellow (`text-yellow-500`, `border-yellow-500`) | Warning, current price, attention |
| Blue (`text-blue-500`, `bg-blue-500`) | Entry, info, neutral highlight |

## Profit/Loss Formatting

Use the `getProfitColor` utility for consistent P/L coloring:

```tsx
import { formatCurrency, getProfitColor } from "@/lib/utils"

<span className={getProfitColor(profit)}>
  {formatCurrency(profit)}
</span>
```

## Cards

Use cards to group related content:

```tsx
<Card>
  <CardHeader>
    <CardTitle className="flex items-center gap-2">
      <Icon className="h-5 w-5" />
      Title
    </CardTitle>
    <CardDescription>Subtitle or description</CardDescription>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

## Loading States

Use `Loader2` with spin animation:

```tsx
import { Loader2 } from "lucide-react"

// In button
<Button disabled={loading}>
  {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
  Submit
</Button>

// Centered loading
<div className="flex items-center justify-center py-8">
  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
</div>
```

## Chart Styling

When using Recharts, use consistent styling:

```tsx
import { Tooltip as ChartTooltip } from "recharts"

<ChartTooltip
  contentStyle={{
    backgroundColor: "hsl(var(--card))",
    border: "1px solid hsl(var(--border))",
    borderRadius: "8px",
  }}
/>
```

Note: Rename recharts `Tooltip` to `ChartTooltip` to avoid conflicts with UI tooltip.

## SMC Terminology

Always provide help tooltips for these terms:

| Term | Explanation |
|------|-------------|
| FVG | Fair Value Gap - price imbalance from rapid moves where price may return |
| OB | Order Block - last candle before strong move, marks institutional entry |
| BOS | Break of Structure - price breaks previous high/low confirming trend |
| CHOCH | Change of Character - first sign of potential trend reversal |
| Liquidity | Areas where stop losses cluster, targets for smart money |
| Mitigation | When price returns to an FVG or OB zone (filled) |

"use client"

import * as React from "react"
import { HelpCircle } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface HelpTooltipProps {
  content: React.ReactNode
  side?: "top" | "right" | "bottom" | "left"
  align?: "start" | "center" | "end"
  className?: string
  iconClassName?: string
}

/**
 * HelpTooltip - A question mark icon that shows helpful information on hover.
 *
 * Usage:
 * ```tsx
 * <HelpTooltip content="This explains what this field does" />
 * ```
 *
 * With custom positioning:
 * ```tsx
 * <HelpTooltip content="Explanation here" side="right" align="start" />
 * ```
 */
export function HelpTooltip({
  content,
  side = "top",
  align = "center",
  className,
  iconClassName,
}: HelpTooltipProps) {
  return (
    <Tooltip delayDuration={200}>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-flex cursor-help text-muted-foreground hover:text-foreground transition-colors",
            className
          )}
        >
          <HelpCircle className={cn("h-4 w-4", iconClassName)} />
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} align={align} className="max-w-xs">
        {content}
      </TooltipContent>
    </Tooltip>
  )
}

interface LabelWithHelpProps {
  children: React.ReactNode
  help: React.ReactNode
  htmlFor?: string
  side?: "top" | "right" | "bottom" | "left"
  className?: string
}

/**
 * LabelWithHelp - A label with an integrated help tooltip icon.
 *
 * Usage:
 * ```tsx
 * <LabelWithHelp help="Explanation of this field" htmlFor="my-input">
 *   Field Name
 * </LabelWithHelp>
 * ```
 */
export function LabelWithHelp({
  children,
  help,
  htmlFor,
  side = "top",
  className,
}: LabelWithHelpProps) {
  return (
    <label
      htmlFor={htmlFor}
      className={cn(
        "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 inline-flex items-center gap-1.5",
        className
      )}
    >
      {children}
      <HelpTooltip content={help} side={side} />
    </label>
  )
}

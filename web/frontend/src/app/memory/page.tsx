"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { getMemoryStats, queryMemory, triggerReflection, getReflectionStatus, getMemoryLessons, deleteMemory } from "@/lib/api"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { RefreshCw, Database, Search, Loader2, Brain, CheckCircle, XCircle, BookOpen, ChevronDown, TrendingUp, TrendingDown, Clock, Award, Trash2 } from "lucide-react"
import { HelpTooltip } from "@/components/ui/help-tooltip"

interface Lesson {
  id: string
  collection: string
  situation: string
  recommendation: string
  tier: "short" | "mid" | "long"
  confidence: number
  outcome_quality: number
  prediction_correct: string
  timestamp: string | null
  reference_count: number
  market_regime?: string
  volatility_regime?: string
}

interface LessonsSummary {
  total: number
  correct_predictions: number
  incorrect_predictions: number
  unknown: number
  avg_confidence: number
  by_tier: Record<string, number>
}

export default function MemoryPage() {
  const [stats, setStats] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  // Query
  const [queryCollection, setQueryCollection] = useState("")
  const [queryText, setQueryText] = useState("")
  const [queryResults, setQueryResults] = useState<any>(null)
  const [querying, setQuerying] = useState(false)

  // Reflection
  const [reflecting, setReflecting] = useState(false)
  const [reflectionResult, setReflectionResult] = useState<any>(null)

  // Lessons
  const [lessons, setLessons] = useState<Lesson[]>([])
  const [lessonsSummary, setLessonsSummary] = useState<LessonsSummary | null>(null)
  const [lessonsLoading, setLessonsLoading] = useState(false)
  const [lessonsTierFilter, setLessonsTierFilter] = useState<string>("all")
  const [lessonsCollectionFilter, setLessonsCollectionFilter] = useState<string>("all")
  const [expandedLessons, setExpandedLessons] = useState<Set<string>>(new Set())

  // Delete confirmation
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [lessonToDelete, setLessonToDelete] = useState<Lesson | null>(null)
  const [deleting, setDeleting] = useState(false)

  const fetchStats = async () => {
    setLoading(true)
    const { data } = await getMemoryStats()
    if (data?.collections) {
      setStats(data.collections)
      if (data.collections.length > 0 && !queryCollection) {
        setQueryCollection(data.collections[0].name)
      }
    }
    setLoading(false)
  }

  const fetchLessons = async () => {
    setLessonsLoading(true)
    const { data } = await getMemoryLessons(
      lessonsCollectionFilter === "all" ? undefined : lessonsCollectionFilter,
      lessonsTierFilter === "all" ? undefined : lessonsTierFilter,
      50
    )
    if (data?.lessons) {
      setLessons(data.lessons)
      setLessonsSummary(data.summary)
    }
    setLessonsLoading(false)
  }

  useEffect(() => {
    fetchStats()
    fetchLessons()
  }, [])

  useEffect(() => {
    fetchLessons()
  }, [lessonsTierFilter, lessonsCollectionFilter])

  const handleQuery = async () => {
    if (!queryCollection || !queryText) return
    setQuerying(true)
    const { data } = await queryMemory(queryCollection, queryText, 10)
    if (data) setQueryResults(data)
    setQuerying(false)
  }

  const handleReflect = async () => {
    setReflecting(true)
    setReflectionResult(null)

    // Start the reflection task (returns immediately with task_id)
    const { data: startData, error: startError } = await triggerReflection()
    if (startError || !startData?.task_id) {
      setReflectionResult({ error: startError || "Failed to start reflection" })
      setReflecting(false)
      return
    }

    const taskId = startData.task_id

    // Poll for completion
    let completed = false
    let attempts = 0
    const maxAttempts = 120 // 4 minutes max (2s intervals)

    while (!completed && attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 2000))
      const { data: statusData, error: statusError } = await getReflectionStatus(taskId)

      if (statusError) {
        setReflectionResult({ error: statusError })
        completed = true
      } else if (statusData?.status === "completed") {
        setReflectionResult(statusData)
        // Refresh stats and lessons after reflection
        fetchStats()
        fetchLessons()
        completed = true
      } else if (statusData?.status === "error") {
        setReflectionResult({ error: statusData.error || "Reflection failed" })
        completed = true
      }
      attempts++
    }

    if (!completed) {
      setReflectionResult({ error: "Reflection timed out - check backend logs" })
    }

    setReflecting(false)
  }

  const toggleLesson = (id: string) => {
    setExpandedLessons(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleDeleteClick = (lesson: Lesson, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent collapsible from toggling
    setLessonToDelete(lesson)
    setDeleteDialogOpen(true)
  }

  const handleConfirmDelete = async () => {
    if (!lessonToDelete) return

    setDeleting(true)
    const { error } = await deleteMemory(lessonToDelete.collection, lessonToDelete.id)

    if (error) {
      console.error("Failed to delete memory:", error)
    } else {
      // Refresh the lessons list
      fetchLessons()
      fetchStats()
    }

    setDeleting(false)
    setDeleteDialogOpen(false)
    setLessonToDelete(null)
  }

  const formatTimestamp = (ts: string | null) => {
    if (!ts) return "Unknown"
    try {
      const date = new Date(ts)
      return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch {
      return ts
    }
  }

  const getTierBadge = (tier: string) => {
    switch (tier) {
      case "long":
        return <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">Long-term</Badge>
      case "mid":
        return <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">Mid-term</Badge>
      default:
        return <Badge variant="secondary">Short-term</Badge>
    }
  }

  const getOutcomeBadge = (prediction: string) => {
    if (prediction === "True") {
      return <Badge variant="buy" className="text-xs"><TrendingUp className="h-3 w-3 mr-1" />Correct</Badge>
    } else if (prediction === "False") {
      return <Badge variant="sell" className="text-xs"><TrendingDown className="h-3 w-3 mr-1" />Incorrect</Badge>
    }
    return <Badge variant="secondary" className="text-xs">Unknown</Badge>
  }

  const totalMemories = stats.reduce((sum, c) => sum + (c.count || 0), 0)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Memory Database</h1>
          <p className="text-muted-foreground">
            Browse lessons learned and query the memory system
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleReflect} disabled={reflecting}>
            {reflecting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Brain className="mr-2 h-4 w-4" />
            )}
            {reflecting ? "Reflecting..." : "Reflect"}
          </Button>
          <Button variant="outline" onClick={() => { fetchStats(); fetchLessons(); }} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Reflection Result */}
      {reflectionResult && (
        <Card className={reflectionResult.error ? "border-red-500/50" : "border-green-500/50"}>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              {reflectionResult.error ? (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  Reflection Failed
                </>
              ) : (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  Reflection Complete
                </>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {reflectionResult.error ? (
              <p className="text-sm text-red-400">{reflectionResult.error}</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Trades Processed</p>
                  <p className="font-medium text-lg">{reflectionResult.trades_processed || 0}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Wins / Losses</p>
                  <p className="font-medium text-lg">
                    <span className="text-green-500">{reflectionResult.winning_trades || 0}</span>
                    {" / "}
                    <span className="text-red-500">{reflectionResult.losing_trades || 0}</span>
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Total P&L</p>
                  <p className={`font-medium text-lg ${(reflectionResult.total_pnl || 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                    {(reflectionResult.total_pnl || 0).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Memories Stored</p>
                  <p className="font-medium text-lg">{reflectionResult.memories_stored || 0}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Tabs for Lessons / Collections / Query */}
      <Tabs defaultValue="lessons" className="space-y-4">
        <TabsList>
          <TabsTrigger value="lessons" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Lessons
          </TabsTrigger>
          <TabsTrigger value="collections" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Collections
          </TabsTrigger>
          <TabsTrigger value="query" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Query
          </TabsTrigger>
        </TabsList>

        {/* Lessons Tab */}
        <TabsContent value="lessons" className="space-y-4">
          {/* Lessons Summary */}
          {lessonsSummary && (
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Total Lessons</CardTitle>
                  <BookOpen className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{lessonsSummary.total}</div>
                  <p className="text-xs text-muted-foreground">
                    {lessonsSummary.by_tier?.long || 0} long-term, {lessonsSummary.by_tier?.mid || 0} mid-term
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
                  <Award className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {lessonsSummary.total > 0
                      ? ((lessonsSummary.correct_predictions / (lessonsSummary.correct_predictions + lessonsSummary.incorrect_predictions)) * 100 || 0).toFixed(0)
                      : 0}%
                  </div>
                  <p className="text-xs text-muted-foreground">
                    <span className="text-green-500">{lessonsSummary.correct_predictions}</span> correct,{" "}
                    <span className="text-red-500">{lessonsSummary.incorrect_predictions}</span> incorrect
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{(lessonsSummary.avg_confidence * 100).toFixed(0)}%</div>
                  <p className="text-xs text-muted-foreground">
                    Based on outcome quality
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Memory Tiers</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="flex gap-2">
                    {getTierBadge("long")}
                    <span className="text-sm">{lessonsSummary.by_tier?.long || 0}</span>
                  </div>
                  <div className="flex gap-2 mt-1">
                    {getTierBadge("mid")}
                    <span className="text-sm">{lessonsSummary.by_tier?.mid || 0}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Lessons Filters */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Lessons Learned
                <HelpTooltip content="Lessons are generated when trades are closed and reflected upon. They capture what worked and what didn't, helping the AI make better decisions in the future." />
              </CardTitle>
              <CardDescription>
                Insights from past trading decisions - these help the AI improve over time
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Filters */}
              <div className="flex gap-4 flex-wrap">
                <div className="space-y-1">
                  <Label className="text-xs">Tier</Label>
                  <Select value={lessonsTierFilter} onValueChange={setLessonsTierFilter}>
                    <SelectTrigger className="w-[140px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Tiers</SelectItem>
                      <SelectItem value="long">Long-term</SelectItem>
                      <SelectItem value="mid">Mid-term</SelectItem>
                      <SelectItem value="short">Short-term</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Collection</Label>
                  <Select value={lessonsCollectionFilter} onValueChange={setLessonsCollectionFilter}>
                    <SelectTrigger className="w-[200px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Collections</SelectItem>
                      {stats.map((coll) => (
                        <SelectItem key={coll.name} value={coll.name}>
                          {coll.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Lessons List */}
              <ScrollArea className="h-[500px]">
                {lessonsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : lessons.length > 0 ? (
                  <div className="space-y-3 pr-4">
                    {lessons.map((lesson) => (
                      <Collapsible
                        key={`${lesson.collection}-${lesson.id}`}
                        open={expandedLessons.has(lesson.id)}
                        onOpenChange={() => toggleLesson(lesson.id)}
                      >
                        <div className="rounded-lg border bg-card">
                          <CollapsibleTrigger className="w-full">
                            <div className="flex items-start justify-between p-4 text-left">
                              <div className="space-y-1 flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  {getTierBadge(lesson.tier)}
                                  {getOutcomeBadge(lesson.prediction_correct)}
                                  <Badge variant="outline" className="text-xs">
                                    {lesson.collection}
                                  </Badge>
                                  {lesson.reference_count > 0 && (
                                    <Badge variant="secondary" className="text-xs">
                                      Referenced {lesson.reference_count}x
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-sm text-muted-foreground line-clamp-2">
                                  {lesson.recommendation.slice(0, 200)}...
                                </p>
                                <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2">
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {formatTimestamp(lesson.timestamp)}
                                  </span>
                                  <span>Confidence: {(lesson.confidence * 100).toFixed(0)}%</span>
                                  {lesson.market_regime && (
                                    <span>Regime: {lesson.market_regime}</span>
                                  )}
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                  onClick={(e) => handleDeleteClick(lesson, e)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                                <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${expandedLessons.has(lesson.id) ? "rotate-180" : ""}`} />
                              </div>
                            </div>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <div className="border-t px-4 py-3 space-y-3">
                              {/* Situation */}
                              <div>
                                <Label className="text-xs text-muted-foreground">Market Situation</Label>
                                <p className="text-sm mt-1 whitespace-pre-wrap bg-muted/50 p-3 rounded-md">
                                  {lesson.situation}
                                </p>
                              </div>
                              {/* Recommendation/Lesson */}
                              <div>
                                <Label className="text-xs text-muted-foreground">Lesson Learned</Label>
                                <p className="text-sm mt-1 whitespace-pre-wrap bg-muted/50 p-3 rounded-md">
                                  {lesson.recommendation}
                                </p>
                              </div>
                              {/* Metadata */}
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                <div>
                                  <Label className="text-xs text-muted-foreground">Confidence</Label>
                                  <p className="font-medium">{(lesson.confidence * 100).toFixed(0)}%</p>
                                </div>
                                <div>
                                  <Label className="text-xs text-muted-foreground">Outcome Quality</Label>
                                  <p className="font-medium">{(lesson.outcome_quality * 100).toFixed(0)}%</p>
                                </div>
                                {lesson.market_regime && (
                                  <div>
                                    <Label className="text-xs text-muted-foreground">Market Regime</Label>
                                    <p className="font-medium">{lesson.market_regime}</p>
                                  </div>
                                )}
                                {lesson.volatility_regime && (
                                  <div>
                                    <Label className="text-xs text-muted-foreground">Volatility</Label>
                                    <p className="font-medium">{lesson.volatility_regime}</p>
                                  </div>
                                )}
                              </div>
                            </div>
                          </CollapsibleContent>
                        </div>
                      </Collapsible>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <BookOpen className="h-12 w-12 mb-4 opacity-50" />
                    <p className="text-lg font-medium">No lessons yet</p>
                    <p className="text-sm">Click "Reflect" after closing trades to generate lessons</p>
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Collections Tab */}
        <TabsContent value="collections" className="space-y-4">
          {/* Stats Overview */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Collections</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.length}</div>
                <p className="text-xs text-muted-foreground">Memory collections</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Total Memories</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalMemories.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">Stored entries</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Largest Collection</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {stats.length > 0 ? (
                  <>
                    <div className="text-lg font-bold">
                      {stats.reduce((max, c) => (c.count > max.count ? c : max), stats[0]).name}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {stats
                        .reduce((max, c) => (c.count > max.count ? c : max), stats[0])
                        .count.toLocaleString()}{" "}
                      entries
                    </p>
                  </>
                ) : (
                  <div className="text-muted-foreground">N/A</div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Collections List */}
          <Card>
            <CardHeader>
              <CardTitle>Collections</CardTitle>
              <CardDescription>All memory collections in the database</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                {stats.length > 0 ? (
                  <div className="space-y-2">
                    {stats.map((coll) => (
                      <div
                        key={coll.name}
                        className="flex items-center justify-between rounded-lg border p-3 hover:bg-muted/50 cursor-pointer"
                        onClick={() => setQueryCollection(coll.name)}
                      >
                        <div className="flex items-center gap-3">
                          <Database className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{coll.name}</span>
                        </div>
                        <Badge variant="secondary">{coll.count.toLocaleString()}</Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    No collections found
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Query Tab */}
        <TabsContent value="query">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Query Memory
              </CardTitle>
              <CardDescription>Search through memories using semantic search</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Collection</Label>
                <Select value={queryCollection} onValueChange={setQueryCollection}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select collection" />
                  </SelectTrigger>
                  <SelectContent>
                    {stats.map((coll) => (
                      <SelectItem key={coll.name} value={coll.name}>
                        {coll.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Search Query</Label>
                <Input
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  placeholder="Enter search query..."
                  onKeyDown={(e) => e.key === "Enter" && handleQuery()}
                />
              </div>

              <Button className="w-full" onClick={handleQuery} disabled={querying || !queryText}>
                {querying ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Search
                  </>
                )}
              </Button>

              {/* Results */}
              {queryResults && (
                <div className="space-y-2">
                  <Label>Results ({queryResults.results?.length || 0})</Label>
                  <ScrollArea className="h-[300px] rounded-md border">
                    {queryResults.results?.length > 0 ? (
                      <div className="p-3 space-y-3">
                        {queryResults.results.map((result: string, i: number) => (
                          <div key={i} className="rounded-lg bg-muted/50 p-3">
                            <p className="text-sm whitespace-pre-wrap">{result}</p>
                            {queryResults.distances?.[i] !== undefined && (
                              <p className="text-xs text-muted-foreground mt-2">
                                Distance: {queryResults.distances[i].toFixed(4)}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex h-full items-center justify-center p-4 text-muted-foreground">
                        No results found
                      </div>
                    )}
                  </ScrollArea>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Memory</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this lesson from the memory database?
              This action cannot be undone.
              {lessonToDelete && (
                <div className="mt-3 p-3 bg-muted rounded-md text-sm">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="outline">{lessonToDelete.collection}</Badge>
                    {getTierBadge(lessonToDelete.tier)}
                  </div>
                  <p className="text-muted-foreground line-clamp-3">
                    {lessonToDelete.recommendation.slice(0, 150)}...
                  </p>
                </div>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

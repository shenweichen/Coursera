import sbt.{Logger, Level}

import collection.mutable.ListBuffer

class GradingFeedback {

  def maxTestScore = maxScore * (1 - styleScoreRatio)

  def maxStyleScore = maxScore * styleScoreRatio

  def totalScore = vTestScore + vStyleScore

  def maxTotalScore = maxTestScore + maxStyleScore

  def feedbackString =
    s"""|${totalGradeMessage(totalScore)}
        |
        |
        |${feedbackSummary.mkString("\n\n")}
        |
        |${feedbackDetails.mkString("\n")}""".stripMargin

  /**
    * `failed` means that there was an unexpected error during grading. This includes
    * - student's code does not compile
    * - our tests don't compile (against the student's code)
    * - crash while executing ScalaTest (not test failures, but problems trying to run the tests!)
    * - crash while executing the style checker (again, not finding style problems!)
    *
    * When failed is `true`, later grading stages will not be executed: this is handled automatically
    * by SBT, tasks depending on a failed one are not run.
    *
    * However, these dependent tasks still fail (i.e. mapR on them is invoked). The variable below
    * allows us to know if something failed before. In this case, we don't add any more things to
    * the log. (see `ProgFunBuild.handleFailure`)
    */
  def isFailed = failed

  /* Methods to build up the feedback log */

  def compileFailed(log: String) {
    failed = true
    addSummary(compileFailedMessage)
    addDetails("======== COMPILATION FAILURES ========")
    addDetails(log)
  }

  def testCompileFailed(log: String) {
    failed = true
    addSummary(testCompileFailedMessage)
    addDetails("======== TEST COMPILATION FAILURES ========")
    addDetails(log)
  }

  def allTestsPassed() {
    addSummary(allTestsPassedMessage)
    vTestScore = maxTestScore
  }

  def testsFailed(log: String, score: Double) {
    failed = true
    addSummary(testsFailedMessage(score))
    vTestScore = score
    addDetails("======== LOG OF FAILED TESTS ========")
    addDetails(log)
  }

  def testExecutionFailed(log: String) {
    failed = true
    addSummary(testExecutionFailedMessage)
    addDetails("======== ERROR LOG OF TESTING TOOL ========")
    addDetails(log)
  }

  def testExecutionDebugLog(log: String) {
    addDetails("======== DEBUG OUTPUT OF TESTING TOOL ========")
    addDetails(log)
  }

  def perfectStyle() {
    addSummary(perfectStyleMessage)
    vStyleScore = maxStyleScore
  }

  def styleProblems(log: String, score: Double) {
    addSummary(styleProblemsMessage(score))
    vStyleScore = score
    addDetails("======== CODING STYLE ISSUES ========")
    addDetails(log)
  }

  def unpackFailed(log: String) {
    failed = true
    addSummary(unpackFailedMessage)
    addDetails("======== FAILURES WHILE EXTRACTING THE SUBMISSION ========")
    addDetails(log)
  }

  def setMaxScore(newMaxScore: Double, newStyleScoreRatio: Double): Unit = {
    maxScore = newMaxScore
    styleScoreRatio = newStyleScoreRatio
  }

  private var maxScore: Double = _
  private var styleScoreRatio: Double = _

  private var vTestScore: Double = 0d
  private var vStyleScore: Double = 0d

  private val feedbackSummary = new ListBuffer[String]()
  private val feedbackDetails = new ListBuffer[String]()

  private var failed = false

  private def addSummary(msg: String): Unit =
    feedbackSummary += msg

  private def addDetails(msg: String): Unit =
    feedbackDetails += msg

  /* Feedback Messages */

  private val unpackFailedMessage =
    """Extracting the archive containing your source code failed.
      |
      |If you see this error message as your grade feedback, please verify that you used the unchanged
      |`sbt submit` command to upload your assignment and verify that you have the latest assignment
      |handout. If you did all of the above and grading still fails, please check the forums to see if
      |this issue has already been reported. See below for a detailed error log.""".stripMargin

  private val compileFailedMessage =
    """We were not able to compile the source code you submitted. This is not expected to happen,
      |because the `submit` command in SBT can only be executed if your source code compiles.
      |
      |Please verify the following points:
      | - You should use the `submit` command in SBT to upload your solution
      | - You should not perform any changes to the SBT project definition files, i.e. the *.sbt
      |   files, and the files in the `project/` directory
      |
      |Take a careful look at the compiler output below - maybe you can find out what the problem is.
      |
      |If you cannot find a solution, ask for help on the discussion forums on the course website.""".stripMargin

  private val testCompileFailedMessage =
    """We were not able to compile our tests, and therefore we could not correct your submission.
      |
      |The most likely reason for this problem is that your submitted code uses different names
      |for methods, classes, objects or different types than expected.
      |
      |In principle, this can only arise if you changed some names or types in the code that we
      |provide, for instance a method name or a parameter type.
      |
      |To diagnose your problem, perform the following steps:
      | - Run the tests that we provide with our hand-out. These tests verify that all names and
      |   types are correct. In case these tests pass, but you still see this message, please post
      |   a report on the forums [1].
      | - Take a careful look at the error messages from the Scala compiler below. They should give
      |   you a hint where your code has an unexpected shape.
      |
      |If you cannot find a solution, ask for help on the discussion forums on the course website.""".stripMargin

  private def testsFailedMessage(score: Double) =
    """The code you submitted did not pass all of our tests: your submission achieved a score of
      |%.2f out of %.2f in our tests.
      |
      |In order to find bugs in your code, we advise to perform the following steps:
      | - Take a close look at the test output that you can find below: it should point you to
      |   the part of your code that has bugs.
      | - Run the tests that we provide with the handout on your code.
      | - The tests we provide do not test your code in depth: they are very incomplete. In order
      |   to test more aspects of your code, write your own unit tests.
      | - Take another very careful look at the assignment description. Try to find out if you
      |   misunderstood parts of it. While reading through the assignment, write more tests.
      |
      |Below you can find a short feedback for every individual test that failed.""".stripMargin.format(score, maxTestScore)

  // def so that we read the right value of vMaxTestScore (initialize modifies it)
  private def allTestsPassedMessage =
    """Your solution passed all of our tests, congratulations! You obtained the maximal test
      |score of %.2f.""".stripMargin.format(maxTestScore)

  private val testExecutionFailedMessage =
    """An error occurred while running our tests on your submission.
      |
      |In order for us to help you, please contact one of the teaching assistants and send
      |them the entire feedback message that you received.""".stripMargin

  // def so that we read the right value of vMaxStyleScore (initialize modifies it)
  private def perfectStyleMessage =
    """Our automated style checker tool could not find any issues with your code. You obtained the maximal
      |style score of %.2f.""".stripMargin.format(maxStyleScore)

  private def styleProblemsMessage(score: Double) =
    """Our automated style checker tool found issues in your code with respect to coding style: it
      |computed a style score of %.2f out of %.2f for your submission. See below for detailed feedback.""".stripMargin.format(score, maxStyleScore)

  private def totalGradeMessage(score: Double) =
    """Your overall score for this assignment is %.2f out of %.2f""".format(score, maxTestScore + maxStyleScore)

}

/**
  * Logger to capture compiler output, test output
  */

object RecordingLogger extends Logger {
  private val buffer = ListBuffer[String]()

  def hasErrors = buffer.nonEmpty

  def readAndClear() = {
    val res = buffer.mkString("\n")
    buffer.clear()
    res
  }

  def clear() {
    buffer.clear()
  }

  def log(level: Level.Value, message: => String) =
    if (level == Level.Error) {
      buffer += message
    }

  // we don't log success here
  def success(message: => String) = ()

  // invoked when a task throws an exception. invoked late, when the exception is logged, i.e.
  // just before returning to the prompt. therefore we do nothing: storing the exception in the
  // buffer would happen *after* the `handleFailure` reads the buffer.
  def trace(t: => Throwable) = ()
}

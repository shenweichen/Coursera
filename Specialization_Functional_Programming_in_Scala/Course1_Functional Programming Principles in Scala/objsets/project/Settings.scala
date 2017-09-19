object Settings {
  val maxSubmitFileSize = {
    val mb = 1024 * 1024
    10 * mb
  }

  val testResultsFileName = "grading-results-log"

  // time in seconds that we give scalatest for running
  val scalaTestTimeout = 850 // coursera has a 15 minute timeout anyhow
  val individualTestTimeout = 240

  // default weight of each test in a GradingSuite, in case no weight is given
  val scalaTestDefaultWeight = 10

  // when students leave print statements in their code, they end up in the output of the
  // system process running ScalaTest (ScalaTestRunner.scala); we need some limits.
  val maxOutputLines = 10 * 1000
  val maxOutputLineLength = 1000

  val scalaTestReportFileProperty = "scalatest.reportFile"
  val scalaTestIndividualTestTimeoutProperty = "scalatest.individualTestTimeout"
  val scalaTestReadableFilesProperty = "scalatest.readableFiles"
  val scalaTestDefaultWeightProperty = "scalatest.defaultWeight"
  val scalaTestReporter = "ch.epfl.lamp.grading.GradingReporter"

}

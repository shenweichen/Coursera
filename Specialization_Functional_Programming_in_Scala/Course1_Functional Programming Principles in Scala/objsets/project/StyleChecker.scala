import sbt.File
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import org.scalastyle._
import com.typesafe.config.ConfigFactory

object StyleChecker {
  val maxResult = 100

  class CustomTextOutput[T <: FileSpec](stream: PrintStream) extends Output[T] {
    // Use the parent class loader because sbt runs our code in a class loader that does not
    // contain the reference.conf file
    private val messageHelper = new MessageHelper(ConfigFactory.load(getClass.getClassLoader.getParent))

    var fileCount: Int = _
    override def message(m: Message[T]): Unit = m match {
      case StartWork() =>
      case EndWork() =>
      case StartFile(file) =>
        stream.print("Checking file " + file + "...")
        fileCount = 0
      case EndFile(file) =>
        if (fileCount == 0) stream.println(" OK!")
      case StyleError(file, clazz, key, level, args, line, column, customMessage) =>
        report(line, column, messageHelper.text(level.name),
          Output.findMessage(messageHelper, key, args, customMessage))
      case StyleException(file, clazz, message, stacktrace, line, column) =>
        report(line, column, "error", message)
    }

    private def report(line: Option[Int], column: Option[Int], level: String, message: String) {
      if (fileCount == 0) stream.println("")
      fileCount += 1
      stream.println("  " + fileCount + ". " + level + pos(line, column) + ":")
      stream.println("     " + message)
    }

    private def pos(line: Option[Int], column: Option[Int]): String = line match {
      case Some(lineNumber) => " at line " + lineNumber + (column match {
        case Some(columnNumber) => " character " + columnNumber
        case None => ""
      })
      case None => ""
    }
  }

  def score(outputResult: OutputResult) = {
    val penalties = outputResult.errors + outputResult.warnings
    scala.math.max(maxResult - penalties, 0)
  }

  def assess(sources: Seq[File], styleSheetPath: String): (String, Int) = {
    val configFile = new File(styleSheetPath).getAbsolutePath

    val messages = new ScalastyleChecker().checkFiles(
      ScalastyleConfiguration.readFromXml(configFile),
      Directory.getFiles(None, sources))

    val output = new ByteArrayOutputStream()
    val outputResult = new CustomTextOutput(new PrintStream(output)).output(messages)

    val msg = s"""${output.toString}
                 |Processed ${outputResult.files}  file(s)
                 |Found ${outputResult.errors} errors
                 |Found ${outputResult.warnings} warnings
                 |""".stripMargin

    (msg, score(outputResult))
  }
}

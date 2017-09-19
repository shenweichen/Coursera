import sbt._

object Courses {

  /** Map of assignmentId -> Assignment details */
  type Assignments = Map[String, Assignment]

  /** Map of courseId -> Assignments */
  type Courses = Map[String, Assignments]

  /** Configurations of the assignments of all courses */
  val all: Courses = Map(
    "progfun1" -> {
        val styleSheetPath = "scalastyle" :: "scalastyle_config.xml" :: Nil
        Map(
          "example" -> Assignment(
            packageName = "example",
            courseraId = CourseraId("g4unnjZBEeWj7SIAC5PFxA", "d5jxI", "xIz9O", None),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath))),
          "recfun" -> Assignment(
            packageName = "recfun",
            courseraId = CourseraId("SNYuDzZEEeWNVyIAC92BaQ", "PzVVY", "Yljln", Some("Ey6Jf")),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath))),
          "funsets" -> Assignment(
            packageName = "funsets",
            courseraId = CourseraId("FNHHMDfsEeWAGiIAC46PTg", "IljBE", "WWsVR", Some("BVa6a")),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath))),
          "objsets" -> Assignment(
            packageName = "objsets",
            courseraId = CourseraId("6PTXvD99EeWAiCIAC7Pj9w", "7hlkb", "d1FGp", Some("Ogg05")),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath)),
            options = Map("grader-timeout" -> "1800")),
          "patmat" -> Assignment(
            packageName = "patmat",
            courseraId = CourseraId("BwkTtD9_EeWFZSIACtiVgg", "2KYZc", "ZjaI7", Some("uctOq")),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath))),
          "forcomp" -> Assignment(
            packageName = "forcomp",
            courseraId = CourseraId("CPJe397VEeWLGArWOseZkw", "v2XIe", "lzaCV", Some("nVRPb")),
            maxScore = 10d,
            styleCheck = Some(StyleCheck(0.2, styleSheetPath)),
            options = Map("grader-timeout" -> "1800"))
        )
    },

    "parprog1" -> {
      Map(
        "example" -> Assignment(
          packageName = "example",
          courseraId = CourseraId("_Cuio9oTEeWUtQpvX4iAkw", "WGx0f", "gM5Y4", None),
          maxScore = 10d
        ),
        "scalashop" -> Assignment(
          packageName = "scalashop",
          courseraId = CourseraId("OpSNmtC1EeWvXAr2bF16EQ", "Q2e1P", "MhXvy", Some("NeGTv")),
          maxScore = 10d
        ),
        "reductions" -> Assignment(
          packageName = "reductions",
          courseraId = CourseraId("lUUWddoGEeWPHw6r45-nxw", "gmSnR", "U1eU3", Some("4rXwX")),
          maxScore = 10d
        ),
        "kmeans" -> Assignment(
          packageName = "kmeans",
          courseraId = CourseraId("UJmFEtoIEeWJwRKcpT8ChQ", "mz8iL", "Olt0g", Some("akLxD")),
          maxScore = 10d
        ),
        "barneshut" -> Assignment(
          packageName = "barneshut",
          courseraId = CourseraId("itfW99oJEeWXuxJgUJEB-Q", "ep95q", "z1ugn", Some("xGkV0")),
          maxScore = 10d
        )
      )
    },

    "progfun2" -> {
      val styleSheetPath = "scalastyle" :: "scalastyle_config.xml" :: Nil
      Map(
        "example" -> Assignment(
          packageName = "example",
          courseraId = CourseraId("lLkU5d7xEeWGkg7lknKHZw", "5QFuy", "AYDPu", None),
          maxScore = 10d,
          styleCheck = Some(StyleCheck(0.2, styleSheetPath))),
        "streams" -> Assignment(
          packageName = "streams",
          courseraId = CourseraId("2iZL1kBCEeWwdxI8PoEnkw", "EKNhX", "VSzXq", Some("Sh2dW")),
          maxScore = 10d,
          styleCheck = Some(StyleCheck(0.2, styleSheetPath)),
          options = Map("grader-timeout" -> "1800", "Xms" -> "512m", "Xmx" -> "512m", "totalTimeout" -> "1500", "individualTimeout" -> "600")),
        "quickcheck" -> Assignment(
          packageName = "quickcheck",
          courseraId = CourseraId("l86W1kt6EeWKvAo5SY6hHw", "DZTNG", "ML01L", Some("DF4y7")),
          maxScore = 10d,
          styleCheck = Some(StyleCheck(0.2, styleSheetPath))
        ),
        "calculator" -> Assignment(
          packageName = "calculator",
          courseraId = CourseraId("QWry5Q33EeaVNg5usvFqrw", "9eOy7", "Qovtr", Some("sO8Cf")),
          maxScore = 10d,
          styleCheck = Some(StyleCheck(0.2, "scalastyle" :: "calculator.xml" :: Nil))
        )
      )

    },

    "bigdata" -> {
      Map(
        "example" -> Assignment(
          packageName = "example",
          courseraId = CourseraId("9W3VuiJREeaFaw43_UrNUw", "vsJoj", "I6L8m", None),
          maxScore = 10d,
          options = Map("Xmx"->"1540m", "grader-memory"->"2048")),
        "wikipedia" -> Assignment(
          packageName = "wikipedia",
          courseraId = CourseraId("EH8wby4kEeawURILfHIqjw", "5komc", "CfQX2", Some("QcWcs")),
          maxScore = 10d,
          options = Map("Xmx"->"1540m", "grader-memory"->"2048", "totalTimeout" -> "900", "grader-cpu" -> "2")),
        "stackoverflow" -> Assignment(
          packageName = "stackoverflow",
          courseraId = CourseraId("7ByAoS4kEea1yxIfJA1CUw", "OY5fJ", "QhzMw", Some("FWGnz")),
          maxScore = 10d,
          options = Map("Xmx"->"1540m", "grader-memory"->"2048", "totalTimeout" -> "900", "grader-cpu" -> "2")),
        "timeusage" -> Assignment(
          packageName = "timeusage",
          courseraId = CourseraId("mVk0fgQ0EeeGZQrYVAT1jg", "y8PO8", "O0akp", Some("T19Ec")),
          maxScore = 10d,
          options = Map("Xmx"->"1540m", "grader-memory"->"2048", "totalTimeout" -> "900", "grader-cpu" -> "2")
        )
      )
    },

    "capstone" -> Map(
      "observatory" -> Assignment(
        packageName = "observatory",
        courseraId = CourseraId("l1U9JXBMEea_kgqTjVyNvw", "CWoWG", "XK71Q", Some("Cr2wv")),
        maxScore = 10d,
        styleCheck = Some(StyleCheck(0.2, "scalastyle" :: "observatory.xml" :: Nil)),
        options = Map("Xmx"->"1600m", "grader-memory"->"2048", "grader-cpu" -> "2")
      )
    )
  )

}

/**
  * @param packageName     Used as the prefix for: (1) handout name, (2) the Scala package, (3) source folder.
  * @param courseraId      Identifies the items and parts of the premium and non-premium assignments.
  * @param maxScore        Maximum score that can be given for the assignment. Must match the value in the WebAPI.
  * @param styleCheck      Configuration of the style checking for the assignment
  * @param options         Options passed to the java process or coursera infrastructure. Following values are
  *                        supported:
  *
  *                            NAME                               DEFAULT              DESCRIPTION
  *                            Xms                                10m                  -Xms for jvm
  *                            Xmx                                256m                 -Xmx for jvm, should less than `grader-memory`
  *                            individualTimeout                  240                  time out of one test case
  *                            totalTimeout                       850                  total time out, should less than `grader-timeout`
  *                            grader-cpu                         1                    number of cpu for coursera infrastructure
  *                            grader-memory                      1024                 memory for coursera infrastructure
  *                            grader-timeout                     1200                 grading timeout for coursera infrastructure
  */
case class Assignment(
  packageName: String,
  courseraId: CourseraId,
  maxScore: Double,
  styleCheck: Option[StyleCheck] = Option.empty,
  options: Map[String, String] = Map.empty
)

/**
  *
  * @param scoreRatio Weight of style checks in the final score, between 0 and 1 (e.g. 0.2 means that style will
  *                   count for 20% of the score)
  * @param styleSheet Path to the scalastyle configuration file (relative from the assignment base directory)
  */
case class StyleCheck(
  scoreRatio: Double,
  styleSheet: List[String]
)

/**
  * Coursera uses two versions of each assignment. They both have the same assignment key and part id but have
  * different item ids.
  *
  * @param key Assignment key
  * @param partId Assignment partId
  * @param itemId Item id of the non premium version
  * @param premiumItemId Item id of the premium version (`None` if the assignment is optional)
  */
case class CourseraId(key: String, partId: String, itemId: String, premiumItemId: Option[String])

---
name: write-script
description: How to write or edit a vertical microdrama show script (content-pipeline/shows/<id>/script.json): episode beats, scenes, shots, blocking, performances, dialogue lines, sound. Use whenever creating a new show or episode, or changing scenes, lines, performances, or character/location descriptions in a script.json.
---

# Writing an episode people keep watching

The schema and what every field means: `packages/shared/src/script.ts` (the
`ShowScript` type and its comments). Read it before writing JSON. Validation:
`pnpm run content:validate`.

The reference episode, `packages/shared/src/reel-short-content-reference/return-of-the-wolf.mp4`, measures as follows:

| Measure | Value |
|---|---|
| Length | 197 s |
| Shots | 81 |
| Median shot | 2.3 s |
| Shots under 1 s | 17 |
| Shots over 5 s | 7 |
| Physical action | roughly half the runtime |
| Spoken lines | about 20 in total, each 2–6 words |

## Craft

1. **A chain, never a list.** Write the beats first, joined only by
   "therefore" or "but", never "and then". Cut any beat that can go without
   breaking the chain. The reference: beasts stampede → therefore the werewolf
   lords run → but a vampire lord blocks them → therefore they change and
   attack → but he breaks all three → therefore the elders send a boy → but
   they call the vampire weak → therefore the vampire mocks the boy → but the
   boy catches his fist → therefore the vampire is thrown into a rock →
   therefore the boy shrugs: "I only beat a weak vampire."

2. **Episode shape (60–90 s):**
   - **Hook, 0–3 s:** danger already in motion, before any face or word (a
     stampede, a scream, a blast, someone running). No calm establishing shot.
   - **Threat proof:** the antagonist wins against someone who matters. The
     cost is visible: a wound, a body thrown, a captive.
   - **Underdog:** the protagonist is dismissed out loud. The status gap is
     said.
   - **Confrontation:** two or three lines of insult, no more.
   - **Reversal:** one physical moment flips status (catches the fist, does not
     flinch, a hidden power). Shown in one shot, never explained.
   - **Payoff:** the humiliation lands (thrown, forced to kneel); onlookers
     react.
   - **Button:** a last line or image that is a punchline or opens a question.

3. **Status, not feelings.** Each scene answers "who is above whom right now,
   and does it flip?". People act on status: mock, threaten, grab, interrupt,
   kneel, laugh. Nobody explains feelings, history, or the rules of the world.

4. **Lines are moves.** 2–8 words, one line per shot: a taunt, threat, insult,
   order, boast, defiance, or punchline.
   - Bad: "The sun-well takes a false bride tonight." (ceremony; nobody does
     anything to anybody).
   - Good: "Kneel, or burn." / "Another puppy came to die?" / "You're nothing."

   A reaction shot (a face, a gasp, a step back) answers every line before the
   next one.

5. **Action over talk.** At least half the scenes have physical action: lunge,
   strike, grab, shove, fall, throw, run, a power going off. A shot of people
   standing and watching exists only as a 1–2 s reaction between moves.

6. **Only what a camera sees.** Every phrase in `action`, `expression`,
   `motion`, and sound names a verb, the body part, and where it goes.
   - Good: "lunges forward and seizes the wrist of {target}", "staggers back
     two steps, a hand on the bleeding chest".
   - Bad: "asserts dominance", "is shaken", "radiates authority".
   - No abstractions (destiny, power, honor) and no metaphors ("like a blade").

7. **Visual text never points at anyone.** Use no names and no pronouns (he,
   him, his, she, her, they, them, it, someone). The video model sees only the
   frame and shows whoever a word points at: "glares down at her", with nobody
   else in the shot, makes it invent a woman.
   - Write the body itself: "the jaw tightens", "lifts the chin".
   - To aim an action at someone or something in the start frame, set `target`
     and write `{target}`. Gaze comes from `lookAtId`.
   - Never mention someone who is not in the shot.

8. **Shot grammar.** Cut between these:
   - a wide (rare, 1–2 s);
   - a group shot or two-shot;
   - a medium close-up for every spoken line;
   - an extreme close-up for the wounding line or a power (eyes, teeth, a hand);
   - inserts of hands, feet, claws, weapons (1–2 s);
   - a low angle on whoever has power, a high angle on whoever is beaten;
   - over-the-shoulder for an exchange;
   - reaction shots.

   Never two consecutive shots with the same framing.

9. **Durations:**
   - insert or reaction: 1–2 s;
   - action beat: 2–3 s;
   - spoken line: 2–4 s (5 s only for a line over ~8 words).

   Most scenes are 2–3 s, so a 60 s episode has about 20–30 scenes.

## The script is a physical world

Every character, landmark, and prop is a mesh that the timeline places. You
never list who is in a shot: previs renders the world from the camera. Who the
video model is told about is what you write: the shot's `subjects` and `over`,
and the `performances`.

- **Coverage is cut from one continuous blocking.** Nobody teleports to
  manufacture a close-up.
- **Give a performance to everyone the model should get right.** Each
  performer is described with their whole look and action; anyone else in the
  frame is not named, and the model draws them as it likes. The shot's `over`
  is described as the shoulder in front. A performer must be in the frame at
  some moment (previs fails otherwise), and a `target` in the start frame.
- **The subjects are the faces.** The reference photos of the shot's `subjects`
  (or, without subjects, of the performers) go to the model, and it paints
  their faces; so list as a subject only someone whose face the shot should
  show, and never the shoulder in front.
- **Change the view through the blocking or the shot type, never through
  coordinates.**
  - Example: the near shoulder fills the edge of an over-the-shoulder shot →
    move that person, or change `side` or `over`.

## Write for what the pipeline can show

- **Every shot starts with people standing**, because the meshes stand
  upright. The motion happens during the clip. Write actions that begin
  standing: lunge, strike, recoil, stagger, fall, turn, grab. Never a shot that
  starts kneeling, lying, or in mid-air.
- **Contact happens inside one clip.** For a grab or a blow, put both people
  within arm's reach at the start of the shot. Give each a performance that
  describes the same contact from their side.
- **The 3D scene gives framing, not acting.**
  - The video model gets where people stand and how big they are; body motion
    and faces come from the words.
  - Keep each shot's motion simple. Keep its camera locked or slow.
- **A speaker with a performance is lip-synced (S2V) from the start still.**
  Their face should be toward the camera in the start frame, best in a medium
  close-up; a speaker without a performance is a voice from off screen. Only one
  face is lip-synced per shot: give a shot one on-screen speaker.
- **Fire, sparks, glow, steam and mist are `effects`,** never words in a
  `body`, attribute or `appearance`: a mesh is built from those words, and a
  flame in a mesh is a solid shape.
- **No text is drawn except the subtitles.** Subtitles are burned in from
  `lines`, and a character with a `name` gets a name card the first time they
  speak on screen.

## Process

1. **Beats:** write the beats as a therefore/but chain (each scene's
   `storyBeat`).
2. **Blocking and shots:** set up the characters, locations, blocking (tracks)
   and shot types, then run `pnpm run content:validate` and
   `pnpm run content:previs`. Fix every error previs reports before writing
   more.
3. **Words:** write the performances, lines and sound.
4. **Check before finishing:**
   - The beats read as a therefore/but chain.
   - The first scene is moving danger, with no line.
   - At least half the scenes are action.
   - Every line is 8 words or fewer, is a move, and is followed by a reaction.
   - Every performance has a verb, a body part, and a direction.
   - No two consecutive scenes share a framing.
   - The last scene is a punchline or a question.

When the benchmark show (`return-of-the-wolf`) is being remade shot for shot,
the reference video decides; these rules do not override it.

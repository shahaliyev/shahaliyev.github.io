---
title: What You Should Know About Concurrency — Part II 
description: Semaphores, Monitors, Message Passing. "Would you believe it, Ariadne? The Monitor scarcely defended himself."
category: [computer science, book summary]
tag: [CPL, concurrency]
---

{% include figcaption.html src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Punishment_sisyph.jpg/677px-Punishment_sisyph.jpg" alt="Semaphores by Titian" caption="Semaphores (1548–49) by <a href='https://en.wikipedia.org/wiki/Titian'>Titian</a>" %}

For the first part, [click here](https://en.wikipedia.org/wiki/The_Myth_of_Sisyphus).

Let’s recall that we should discuss three methods to provide mutually exclusive access to a resource **(semaphores, monitors, message passing)**. However, there are definitely more options. The following links lead to their corresponding Wikipedia pages.

*   [Locks](https://en.wikipedia.org/wiki/Lock_%28computer_science%29 "Lock (computer science)") (mutexes)
*   [Readers–writer locks](https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock "Readers–writer lock")
*   [Recursive locks](https://en.wikipedia.org/wiki/Reentrant_mutex "Reentrant mutex")
*   [Semaphores](https://en.wikipedia.org/wiki/Semaphore_%28programming%29 "Semaphore (programming)")
*   [Monitors](https://en.wikipedia.org/wiki/Computer_monitor)
*   [Message passing](https://en.wikipedia.org/wiki/Message_passing "Message passing")
*   [Tuple space](https://en.wikipedia.org/wiki/Tuple_space "Tuple space")

{% include toc.html %}

## Semaphores

In 1965 (or in 1962, or in 1963), famous Dutch computer scientist [**Edsger W. Dijkstra**](https://en.wikipedia.org/wiki/Edsger_W._Dijkstra)[^1] proposed the usage of semaphores as a way of providing exclusive access to shared resources. **Semaphore** is simply a variable that is used for the synchronization of tasks. It’s a guard, it either allows access or not — just like the green traffic light.

With the help of semaphores, let’s solve the [**producer-consumer problem**](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem#:~:text=In%20computing%2C%20the%20producer%E2%80%93consumer,buffer%20used%20as%20a%20queue.) that we have discussed in the previous part. As a quick reminder, we have a buffer with limited space, and we should not allow the producer to put any data into it if the buffer is full, and at the same time, not allow the consumer to take any data from it if the buffer is empty.

Initially, we need to understand two important subprograms, called _wait_ and _release_[^2] (refer to the notes below for a free Dutch vocabulary lesson).

**Wait** takes semaphore as a parameter and checks for its value. If the semaphore’s value is greater than zero, then the needed operation can be freely carried out. But before that, the counter value of the semaphore is decremented to make it take note that less space is remaining in the buffer. If the semaphore is zero, the operation is put in the queue.

Let’s take the following example for allegedly easier understanding. Initially, there is room for 10 books (resources) on the shelf (buffer).


{% include figcaption.html src="/assets/img/concurrency/semaphore-1.png" alt="Semoaphore demonstration" %}

We initialize a semaphore called _emptyspacesabandonedplaces_ to 10. If we want to put one book on the shelf, we must check if there is space. Therefore, as we are machine and not human, we run the subprogram _wait,_ and see that our semaphore equals 10. It means there are 10 empty places on the shelf. Before putting the book on the shelf, we decrement the semaphore by 1 to indicate that we will soon have 9 empty spaces.

{% include figcaption.html src="/assets/img/concurrency/semaphore-2.png" alt="Semoaphore demonstration" %}

If the shelf is completely filled with books, our semaphore will be equal to zero (zilch space). In which case, we put the book in the queue (e.g. on the table next to the shelf).


{% include figcaption.html src="/assets/img/concurrency/semaphore-3.png" alt="Semoaphore demonstration" %}

**Release** subprogram. If there are no books in the queue waiting to be put on the shelf, it increments the semaphore (e.g. there are again 10 empty places on the shelf). If there are books waiting in the queue, then the release subprogram moves one of the books on the table to the **task-ready queue** (refer to the task states section in [Part I](/posts/concurrency-1) for more information). With our analogy, we simply grab the book from the table and hope to put it on the shelf.

Before writing a simplified **pseudo-code**, we must take into account the competition synchronization. For this, we will use **access semaphore** for mutually exclusive access to buffer (e.g. two producers will not be able to add items to the empty queue at the same time).

There are a couple of **important features** to note down about semaphores.

Firstly, two types of semaphores exist: **binary** and **counting**. In the above example, we used counting semaphores. A binary semaphore can obviously be 1 or 0 depending on whether a condition is met or not.

Secondly, operations in subprograms must take place **atomically**. That is, all the operations must be completed in one go with no external interference, as otherwise, subprograms may easily “forget” to increment/decrement semaphores. Atomicity can be achieved on hardware possessing [read-modify-write](https://en.wikipedia.org/wiki/Read-modify-write "Read-modify-write") instruction (most computers are provided with it). In the absence of it, a [software mutual exclusion algorithm](https://en.wikipedia.org/wiki/Mutual_exclusion#Software_solutions "Mutual exclusion") is used.

And lastly, according to [Per Brinch Hansen](https://en.wikipedia.org/wiki/Per_Brinch_Hansen), semaphores are ideal only for ideal programmers who never make mistakes. Indeed, the pseudo-code we have just seen is as simple as it gets. In reality, a programmer must be careful not to forget any relevant steps and must maintain the integrity of a program.

Although, as noted in [Wikipedia](https://en.wikipedia.org/wiki/Semaphore_%28programming%29), even in that case multi-resource deadlock can occur (different resources are managed by different semaphores, and tasks need to use more than one resource simultaneously).

{% include figcaption.html src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/George_Frederic_Watts_-_The_Minotaur_-_Google_Art_Project.jpg/616px-George_Frederic_Watts_-_The_Minotaur_-_Google_Art_Project.jpg" alt="The Monitor by George F. Watts" caption="The Monitor (1885) by <a href='https://en.wikipedia.org/wiki/George_Frederic_Watts'>George F. Watts</a>" %}

For short stories, always refer to [Borges](https://en.wikipedia.org/wiki/The_House_of_Asterion), Maupassant, and Chekhov.

## Monitors

As we have some imagination about the shortcomings of semaphores, we can now deal with monitors. In 1971, Edsger Dijkstra suggested the encapsulation of all synchronization operations. After two years, **Per Brinch Hansen**, whose Wikipedia page we have already read, formalized this idea, and [Sir Anthony Hoare](https://en.wikipedia.org/wiki/Tony_Hoare) defined it as monitors in 1974.

An essential feature of monitors is that shared data is located within the monitor rather than in a client unit. That makes monitors powerful in comparison to semaphores during **competition synchronization**. However, **cooperation synchronization** should still be implemented by a programmer, and especially, possible underflow/overflow cases must be taken care of.

{% include figcaption.html src="/assets/img/concurrency/monitor.png" alt="Usage of a Monitor in a Program. From R. W. Sebesta’s book, re-sketched" caption="Usage of a Monitor in a Program, re-sketched from <a href='https://www.amazon.com/Concepts-Programming-Languages-Robert-Sebesta/dp/0134997182'>R. W. Sebesta’s book</a>." %}

Monitor is an [**abstract data type**](https://en.wikipedia.org/wiki/Abstract_data_type#:~:text=In%20computer%20science%2C%20an%20abstract,the%20behavior%20of%20these%20operations.) **(ADT)**.

Different programming languages provide different ways of implementing monitors. In C# we have a predefined Monitor class. Whereas in Java monitors can be implemented in a class designed as an ADT, with the shared data being _type_. Ada 83 includes a general tasking model to support monitors, whereas Ada 95 has **protected objects,** when both approaches use message passing to support concurrency (Sebesta). Ada 83 supports only synchronous message passing which we will talk about pretty soon.

The Wikipedia link for Monitor, this time the real one: [https://en.wikipedia.org/wiki/Monitor\_(synchronization)](https://en.wikipedia.org/wiki/Monitor_%28synchronization%29)

## Message Passing

{% include figcaption.html src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg/640px-Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg" alt="Message Passing" caption="Message Passing (1508-12) by <a href='https://en.wikipedia.org/wiki/Michelangelo'>Michelangelo</a>" %}

Couldn’t find any normal painting of [Hermes](https://en.wikipedia.org/wiki/Hermes), therefore this.

Message Passing (unrelated to the OOP concept) was developed by **Brinch Hansen and Hoare** in 1978. [Guarded commands](https://en.wikipedia.org/wiki/Guarded_Command_Language) are the basis of the construct designed for message passing (Sebesta).

Message Passing can be synchronous or asynchronous. [The book of Sebesta](https://www.amazon.com/Concepts-Programming-Languages-Robert-Sebesta/dp/0134997182) talks about synchronous message passing only, so we will begin with that.

In synchronous message passing, busy tasks cannot be interrupted by other units. Here, a task can notify other tasks that it completed its processing and is ready to receive messages. A task can also suspend its execution, either due to being idle or waiting for some expected information from another unit.

When a task waits for a message, and at the same time, another task sends that message, the possible message transmission is called **rendezvous** (pronounced as “rehn-dez-voh-ooz”). [^3] Rendezvous can happen if both parties want it to happen, and it can be transmitted in either one or both directions.

A fantastic explanation of Message Passing can be found at [Bartosz Milewski’s blog](https://bartoszmilewski.com/2009/02/10/message-passing-sync-or-async/). Below, for example, is the difference between synchronous and asynchronous message passing taken from the mentioned post.

> In the **synchronous** model, the sender blocks until the receiver picks up the message. The two have to rendezvous.

> In the **asynchronous** model, the sender drops the message and continues with its own business.

After reading the article, I got the impression that the synchronous model is right for the academic explanation, whereas the asynchronous model is more practical. Just like [Scheme, as opposed to Common Lisp](/posts/lisp).

## Footnotes

[^1]: If unsure who invented something, say “Dijkstra”.

[^2]: In Edsger Dijkstra’s original paper, he uses **V** and **P** as the initials for _verhogen_, meaning “increase”, and _probeer te verlagen_, meaning “try to reduce”. English textbooks introduce various translations for that: _up and down, signal and wait, release and acquire, post and pend._ But as we are obliged to confuse ourselves, we should go with Robert W. Sebesta’s mixed _release and wait_ option.

[^3]: Joking